using System;
using System.Collections.Generic;
using System.Text;
using System.Runtime.InteropServices;
using StructureInterfaces;
using System.Diagnostics;
using System.Data.SqlClient;
using System.Data;
using MPI;
using System.Runtime.Serialization;
using Utilities;

[assembly: CLSCompliant(true)]

namespace KDTreeStructures
{
    /// <summary>
    /// This is the KD Tree class.
    /// </summary>
    [Serializable()]
    public sealed unsafe class KDTree : ITree, ISerializable
    {
        /**
         * Debugging members.
         */
        
        private DateTime startTime = DateTime.Now;
        private DateTime endTime = DateTime.Now;

        /**
         * Static members
         */

        private static String CREATE_TREE_TABLE =
            " Create Table <TreeTableName> ( " +
            " NodeId int, SplitDimension int, SplitPoint float(53), " +
            " Level int, NumPoints int, LC int, RC int, " +
            " MinBoxValues nvarchar(max), MaxBoxValues nvarchar(max) )";
        /*
        private static String INSERT_INTO_MASTER =
            " INSERT INTO Master.dbo." + SQLInterface.MASTER_TABLE_NAME +
            " Values ( '<DataDb>', '<DataTable>' ,'<DatabaseName>', '<TreeTableName>', '<MappingTableName>' ,'KD', <k> , <fanOut>, <nNodes>, '<TreeSpecifics>' ) ";
        */
        
        /**
         * Data Members
         */
        public int k;                           // dimensionality
        private const int stdlLeafSize = 30;    // the size below which node is made leaf
        private const int fanOut = 2;           // number of children of a node
        private int nNodes;                     // number if nodes in the tree
        private int nPoints;                    // total number of data points
        public List<KNode> nodeList;            // the list of nodes
        private double[] boxMins;               // the hyper rectangle lower and upper bounds ( nNodes * k )
        private double[] boxMaxs;               // these may be larger but only use upto nNodes *k
        private int[] nodeDataMapping;          // for data with ids[i],nodeDataMapping[i] is the node id it belongs to 
        private int* idBegin;                   // the starting address of the ids
        private int[] ids;                      // the ids of the data, held so that they can be stored.
        private double* minPtr, maxPtr;         // these are used to iterate through boxMins & boxMaxs
        private int maxMedianSplitLevel = -1;   // the level upto which median region splitting is employed
        private List<KNode> globalNodeList;     // this is only used to create the global tree
        private double[] globalBoxMins;
        private double[] globalBoxMaxs;

        public KDTree() { } // default constructor

        // Deserialization function
        public KDTree(SerializationInfo info, StreamingContext ctxt)
        {
            //Get the values from info and assign them to the appropriate properties
            //k = (int)info.GetValue("Dimensionality", typeof(int));
            nNodes = (int)info.GetValue("nNodes", typeof(int));
            //nPoints = (int)info.GetValue("nPoints", typeof(int));
        }
        
        //Serialization function.
        public void GetObjectData(SerializationInfo info, StreamingContext ctxt)
        {
            //info.AddValue("Dimensionality", k);
            info.AddValue("nNodes", nNodes);
            //info.AddValue("nPoints", nPoints );
        }

        /**
         * Following are the MPI variables
         */
        private int MPIRank;
        private int MPIWorldSize;
        private Communicator MPIWorld;

        private void InitMPIVars()
        {
            MPIWorld = Communicator.world;
            MPIRank = Communicator.world.Rank;
            MPIWorldSize = Communicator.world.Size;
            maxMedianSplitLevel = (int)Math.Log(MPIWorldSize, 2);
        }

        /// <summary>
        /// This function reaches the root of the subtree to be built by this
        /// task. In a trivial form assume only 2 tasks are running at a time.
        /// One task builds the left subtree and the other the right subtree.
        /// This can recursively be extended for more tasks.
        /// </summary>
        private unsafe void IdentifyLocalWorkingSet(ref int[] ids, ref double[] data)
        {
            double[] minData = new double[k];
            double[] maxData = new double[k];

//            Queue<double*> dataPointers = new Queue<double*>();
            int globalListSize = MPIWorldSize - 1;
            globalNodeList = new List<KNode>(globalListSize);
            globalBoxMins = new double[globalListSize*k];
            globalBoxMaxs = new double[globalListSize*k];

            fixed (double* dp = data, minP = minData, maxP = maxData )
            {
                minPtr = minP;
                maxPtr = maxP;
                double* dataPtr = dp;
                fixed (int* ip = ids)
                {
                    int* idPtr = ip;
                    int seperateLevel = (int)Math.Log(MPIWorldSize, 2);
                    int dataSize = ids.Length;
                    int rankShifter = MPIRank;
                    int stride = MPIWorldSize;
                    //int globalBoxIdx = 0;
                    int lvl = 0;
                    
                    //Print("Seperate Level = " + seperateLevel);
                    for (int i = 0; i < seperateLevel; i++)
                    {
                        double splitPoint;  // temporarily set to the mid point
                        int splitDimension = GetMaxRangeDimension(dataPtr, idPtr, dataSize, out splitPoint);
                        int leftSize;
                        bool splitOccured;
                        splitOccured = DoMedianSplit(
                            dataPtr, idPtr, dataSize, splitDimension, out splitPoint, out leftSize);
                        
                        int rightSplitSize = dataSize - leftSize;

                        // create globla nodes
                        for (int j = 0; j < MPIWorldSize && stride > 0; j += stride)
                        {
                            
                            if (MPIRank == j)
                            {
                                // add a node to your global list
                                KNode node = new KNode(splitDimension, splitPoint,
                                    nNodes + 1, nNodes, leftSize + rightSplitSize, lvl, this);
                                if (MPIRank == 4)
                                    Print("split point : " + splitPoint);
                                globalNodeList.Add(node);
                                Array.Copy(minData, 0, globalBoxMins, nNodes*k, k);
                                Array.Copy(maxData, 0, globalBoxMaxs, nNodes*k, k);
                                nNodes++;
                                node.SetLC(nNodes);

                            }
                            
                        }
                        stride /= 2;
                        lvl++;

                        if (splitOccured && leftSize != 1 && rightSplitSize != 1)
                        {
                            rankShifter = MPIRank;
                            rankShifter = rankShifter >> seperateLevel - i - 1;
                            if (rankShifter % 2 == 1)   // go to the right sub part
                            {
                                dataPtr += leftSize * k;
                                idPtr += leftSize;
                                dataSize = rightSplitSize;
                            }
                            else
                            {
                                // keep to the left sub part
                                dataSize = leftSize;
                            }
                            
                        }
                        else
                        {
                            throw new Exception();
                        }

                    }
                    // create new ids and data so that the memeory can be reused efficiently
                    ids = new int[dataSize];
                    data = new double[dataSize * k];
                    for (int i = 0; i < dataSize * k; ++i)
                    {
                        data[i] = *dataPtr++;
                    }
                    for (int i = 0; i < dataSize; ++i)
                    {
                        ids[i] = *idPtr++;
                    }
                }
                
            }
            //PrintGlobalList();
            //Print("data size: " + ids.Length);
        }

        private void PrintGlobalList()
        {
            for (int i = 0; i < globalNodeList.Count; i++)
            {
                Print("Node id:" + globalNodeList[i].GetNodeId() + " num points " + globalNodeList[i].GetNumPoints());
            }
        }

        private void Print(String str)
        {
            Console.WriteLine("Rank " + MPIRank + ": " + str);
        }

        public unsafe void InitializeTree(int[] ids, double[] data)
        {
            MPIInitializeTree(ids, data);
        }


        /// <summary>
        /// This method creates the tree from the input data. 
        /// The assumption here is that the data is normalized. 
        /// Currently we do not worry about overflowing the initial size 
        /// but that has to be taken care of. 
        /// </summary>
        /// <param name="data">The data used to create the tree</param>
        public unsafe void MPIInitializeTree(int[] ids, double[] data)
        {
            // intitialize the MPI variable
            InitMPIVars();
            k = data.Length / ids.Length;
            nNodes = 0;
            //Print("here 0");
            IdentifyLocalWorkingSet(ref ids, ref data);
            //nNodes = 0;
            // setup the class data variables
            this.ids = ids;
            nPoints = ids.Length;
            //Trace.WriteLine("KDTree.InitializeTree(Datum[]): Set the tree dimensionality to " + k);
            int initialArraySize = GetInitialArraySize(ids.Length);
            //Trace.WriteLine("KDTree.InitializeTree(Datum[]): DataLength is "
            //    + ids.Length + " and initial node array size set to "
            //    + initialArraySize);

            System.GC.Collect();
            nodeList = new List<KNode>(initialArraySize);
            boxMins = new double[initialArraySize * k];
            boxMaxs = new double[initialArraySize * k];
            nodeDataMapping = new int[nPoints];
            Trace.WriteLine("here 1");
            Trace.Flush();
            fixed (double* dataP = data, minP = boxMins, maxP = boxMaxs)
            {
                fixed (int* idP = ids)
                {
                    idBegin = idP;
                    minPtr = minP;
                    maxPtr = maxP;
                    
                    Trace.WriteLine("KDTree.InitializeTree(Datum[]): Calling CreateTreeRecursively().");
                    /**
                     * Create tree recursively
                     */
                    startTime = DateTime.Now;
                    CreateTreeRecursively(dataP, idP, ids.Length, (int)Math.Log(MPIWorldSize, 2));
                    Trace.WriteLine("here 2");
                    Trace.Flush();
            
                    endTime = DateTime.Now;
                }
            }
            Trace.WriteLine("KDTree.InitializeTree(Datum[]): Finished Creating Tree. Time taken- " + PrintTime(startTime, endTime));
            Trace.WriteLine("No. of Nodes: " + nNodes);

            /*
            for (int i = 0; i < 1000000; ++i)
            {
                int id = ids[i];
                if( id != idcopy[id-1] )
                    Trace.WriteLine("fuck");
                if (data[i * k + 0] != dataCopy[(id - 1) * k + 0]
                    || data[i * k + 1] != dataCopy[(id - 1) * k + 1]
                    || data[i * k + 2] != dataCopy[(id - 1) * k + 2]
                    || data[i * k + 3] != dataCopy[(id - 1) * k + 3]
                    || data[i * k + 4] != dataCopy[(id - 1) * k + 4])
                    Trace.WriteLine("fuck");

            }
            Trace.WriteLine("wrijksdhsjk");
             */

            // local trees have finished ... now to assemble the main tree
            //Print("Number of Nodes = " + nNodes);
            //Print("Global node size = " + globalNodeList.Count);
            //Print("subtree node size = " + nodeList.Count);
            
            CollateSubtrees();
            //DumpDataAndIdsIntoTable(ids,data);

        }

        private void CheckStatus(CompletedStatus status)
        {
            if (status.Cancelled)
            {
                throw new Exception("Couldn't Recieve data");
            }
        }

        /// <summary>
        /// Once the local subtrees have been build the global subtree must be built.
        /// Also, each task must update its local subtree to reflect its position in
        /// the global subtree. Essentially certain node properties must change such
        /// as BoxIdx, NodeId, LC index, RC index etc.
        /// </summary>
        private void CollateSubtrees()
        {
            Print("Number of nodes in subtree is " + nodeList.Count);
            //Print("subtree root node value is " + nodeList[0].GetNodeId());
            globalNodeList.Add(nodeList[0]);
            //Print("added node to global list" + nodeList[0].GetNodeId() + " size " + nodeList[0].GetNumPoints());
            globalNodeList.TrimExcess();
            List<int> globalNodeNumNodes = new List<int>(globalNodeList.Count);
            //Print("adding the subtree sizes : " + nodeList.Count);
            globalNodeNumNodes.Add(nodeList.Count);

            /**
             * In this part the following sequence is generated. (0,1),(2,3),(4,5)...
             * then (0,2),(4,6)... the idea here is that 0,2,4,6 etc.. build some nodes
             * that for part of the global tree. 0 makes Log N number of global nodes. 4
             * makes (Log N) -1 and so on.
             */
            int levels = (int)Math.Log(MPIWorldSize, 2);
            int stride = 2;
            for (int i = 0; i < levels; i++)
            {
                for (int j = 0; j < MPIWorldSize; j += stride)
                {
                    //send and recieve j and j + (stride/2)
                    CompletedStatus status;
                    if (MPIRank == j)
                    {
                        //Print("Sending to " + (j + (stride / 2)));
                        int siblingNNodes = MPIWorld.Receive<int>(j + (stride / 2), 0);
                        globalNodeNumNodes.Insert(0, siblingNNodes + globalNodeNumNodes[0] + 1);
                    }
                    else if(MPIRank == j + (stride / 2))
                    {
                        //Print("Recieving from " + j );
                        // send the number of nodes you have 
                        MPIWorld.Send<int>(globalNodeNumNodes[0] , j, 0);
                    }
                }
                stride *= 2;
            }

            

            /**
             * Now we generate values (0,4), then (0,2),(4,6)...
             * Where the final values are passed on down to the subtrees
             * for update.
             */ 
            stride = MPIWorldSize;
            int curr = 0;
            for (int i = 0; i < levels; i++)
            {
                // create globla nodes
                for (int j = 0; j < MPIWorldSize && stride > 0; j += stride)
                {
                    if (MPIRank == j)
                    {
                        curr++;
                        //Print("Send to " + (j + (stride / 2)) + " add value " + (globalNodeNumNodes[curr] + globalNodeList[curr].GetNodeId()));
                        MPIWorld.Send<int>(globalNodeNumNodes[curr] + globalNodeList[curr].GetNodeId() - 1, j + (stride/2), 0);
                        int siblingNodeId = MPIWorld.Receive<int>(j + (stride / 2), 0);
                        globalNodeList[curr - 1].SetRC(siblingNodeId - 1);
                    }
                    else if (MPIRank == j + (stride / 2))
                    {
                        //Print("Recv from " + (j));
                        int leftSiblingSize = MPIWorld.Receive<int>(j, 0);
                        UpdateNode(globalNodeList, leftSiblingSize);
                        MPIWorld.Send<int>(globalNodeList[curr].GetNodeId(), j, 0);
                    }

                }
                stride /= 2;
            }
            // now update the subtrees
            if (MPIRank != 0)
            {
                int addValue = globalNodeList[globalNodeList.Count - 1].GetNodeId() - globalNodeList.Count;
                //Print("Now adding to subtree value " + globalNodeList[globalNodeList.Count - 1].GetNodeId());
                /**
                 * We start from one because the root of the subtree had already been inserted into the
                 * global list and is therefore updated already.
                 */
                nodeList[0] = globalNodeList[globalNodeList.Count-1];
                for (int i = 1; i < nodeList.Count; i++)
                {
                    nodeList[i].SetNodeId(nodeList[i].GetNodeId() + addValue);
                    if (!nodeList[i].IsLeaf())
                    {
                        nodeList[i].SetLC(nodeList[i].GetLC() + addValue);
                        nodeList[i].SetRC(nodeList[i].GetRC() + addValue);
                    }
                    nodeList[i].SetBoxIdx(nodeList[i].GetBoxIdx() + addValue);
                }
            }
            // finally delete the root of the subtree from the global list
            globalNodeList.RemoveAt(globalNodeList.Count - 1);
            globalNodeNumNodes.RemoveAt(globalNodeNumNodes.Count - 1);
            //nodeList.AddRange(globalNodeList);

            PrintGlobalNodeId();
            Print(" Root of subtree has the node id " + nodeList[0].GetNodeId());
            Print(" sub tree size " + nodeList.Count);
            //Print("Subtree root node id = " + nodeList[0].GetNodeId());
        }

        private void PrintGlobalNodeId()
        {
            for( int i =0 ;i < globalNodeList.Count; i++ )
            {
                Print(""+globalNodeList[i].GetNodeId());
            }
        }

        private void PrintTheseSizes(List<int> sizes)
        {
            for (int i = 0; i < sizes.Count; i++)
                Print(sizes[i] + ", ");
        }

        private void UpdateNode(List<KNode> global, int value)
        {

            for (int i = 0; i < global.Count; i++)
            {
                //Print("Changing value " + global[i].GetNodeId() + " adding " + value);
                global[i].SetNodeId(global[i].GetNodeId() + value );
                global[i].SetLC(global[i].GetLC() + value);
                global[i].SetRC(global[i].GetRC() + value);
                global[i].SetBoxIdx(global[i].GetBoxIdx() + value);
            }
            //node.SetLC(node.GetLC() + value + 1);
        }

        /*
        public void DumpDataAndIdsIntoTable( int[] ids, double[] data )
        {
            SqlConnection connection = new SqlConnection();
            //connection.ConnectionString = "Context Connection=true";
            connection.ConnectionString = "Data Source=KLEENE; Initial Catalog="
                + "nntest" + "; Integrated Security=True;";
            connection.Open();
            // first save the table information in master needed?
            SqlCommand comm = new SqlCommand();
            comm.Connection = connection;
            String insert = "insert into testOrder values(";
            
            for( int i = 0 ; i < 1000000 ; ++i )
            {
                string temp = insert;
                temp += ids[i] + ", ";
                for( int j = 0 ; j < k - 1 ; j++ )
                {
                    temp += data[i * k + j] + ", ";
                }
                temp += data[i * k + k - 1] ;
                temp += " ) ";
                comm.CommandText = temp;
                if (i % 10000 == 0)
                    Console.WriteLine(temp);
                comm.ExecuteNonQuery();
            }
            connection.Close();
        }
        
        */

        private unsafe void UpdateNodeDatumMapping(int* idPtr, int nodeId, int dataSize)
        {
            long idx = (idPtr - idBegin);
            for (long i = idx; i < idx + dataSize; ++i)
            {
                nodeDataMapping[i] = nodeId;
            }
        }

        private unsafe void CreateTreeRecursively(double* dataPtr, int* idPtr, int dataSize, int level)
        {

            // check if this node should be pos1 leaf
            if (dataSize <= stdlLeafSize)
            {
                // is leaf ... create this as pos1 leaf node and then return
                SetBoundedRegion(dataPtr, dataSize);
                nodeList.Add(new KNode(-1, -1, nNodes + 1, nNodes, dataSize, level, this));
                UpdateNodeDatumMapping(idPtr, nNodes + 1, dataSize);
                nNodes++;
                minPtr += k;
                maxPtr += k;
                return;
            }

            // otherwise split this data into 2. first find where to split
            double splitPoint;  // temporarily set to the mid point
            int splitDimension = GetMaxRangeDimension(dataPtr, idPtr, dataSize, out splitPoint);
            //Console.WriteLine("split dimension: " + splitDimension);
            minPtr += k;
            maxPtr += k;
            // now split the data based on either midpoint or median
            int leftSize;
            bool splitOccured;
            if (level > maxMedianSplitLevel)
            {
                // do the mid point split
                splitOccured = GetLeftRightSplit(
                    dataPtr, idPtr, dataSize, splitDimension, splitPoint, out leftSize);
            }
            else
            {
                // do the median split
                splitOccured = DoMedianSplit(
                    dataPtr, idPtr, dataSize, splitDimension, out splitPoint, out leftSize);
                //Console.WriteLine("done once split point: " + splitPoint + " dimension " + splitDimension + " leftsize " + leftSize );
                //Console.ReadLine();
            }

            int rightSplitSize = dataSize - leftSize;

            if (splitOccured && leftSize != 1 && rightSplitSize != 1)
            {

                // then this is not pos1 leaf node
                int nodeIndex = nNodes;
                KNode node = new KNode(splitDimension, splitPoint, nNodes + 1, nNodes, dataSize, level, this);
                nodeList.Add(node);
                nNodes++;
                level++;
                node.SetLC(nNodes);
                CreateTreeRecursively(dataPtr, idPtr, leftSize, level);
                node.SetRC(nNodes);
                CreateTreeRecursively((dataPtr + (leftSize * k)), idPtr + leftSize, rightSplitSize, level);
            }
            else
            {
                nodeList.Add(new KNode(-1, -1, nNodes + 1, nNodes, dataSize, level, this));
                UpdateNodeDatumMapping(idPtr, nNodes + 1, dataSize);
                nNodes++;
            }
        }

        private int GetInitialArraySize(int dataSize)
        {
            int baseTwo = (int)Math.Ceiling(Math.Log(

                                    ((double)(2 * dataSize) / stdlLeafSize), 2));
            return (int)(2 * Math.Pow(2, baseTwo));
        }

        public unsafe int GetMaxRangeDimension(double* dataPtr, int* idPtr, int dataSize, out double midPoint)
        {
            SetBoundedRegion(dataPtr, dataSize);
            double maxRange = (*maxPtr - *minPtr);
            int maxRangeIndex = 0;
            for (int i = 1; i < k; ++i)
            {
                double range = *(maxPtr + i) - *(minPtr + i);
                if (range > maxRange)
                {
                    maxRange = range;
                    maxRangeIndex = i;
                }
            }
            midPoint = (*(maxPtr + maxRangeIndex) + *(minPtr + maxRangeIndex)) / 2;
            //Console.WriteLine("SplitPoint:" + midPoint);
            //Console.WriteLine("Splitdimension:" + maxRange);
            return maxRangeIndex;
        }

        /// <summary>
        /// This function simply scans the data starting at dataPtr an 
        /// k*dataSize elements and sets the boxMin's and boxMax's
        /// arrays using the global minPtr, maxPtr pointers. Note:
        /// It does not increment the pointers. That is the responsibility
        /// of the calling functions.
        /// </summary>
        /// <param name="dataPtr"></param>
        /// <param name="dataSize"></param>
        private unsafe void SetBoundedRegion(double* dataPtr, int dataSize)
        {
            double* p = dataPtr;
            double* mn = minPtr;
            double* mx = maxPtr;
            for (int i = 0; i < k; ++i)
            {

                *(mx + i) = *p;
                *(mn + i) = *p;
                p++;
            }
            for (int i = 1; i < dataSize; ++i)
            {
                for (int j = 0; j < k; ++j)
                {
                    if (*(mn + j) > *p)
                    {

                        *(mn + j) = *p;
                    }
                    else if (*(mx + j) < *p)
                    {

                        *(mx + j) = *p;
                    }
                    p++;
                }
            }

        }
        /*
        private unsafe void PrintData(int* id, double* data)
        {
            Trace.Write("Id: " + *id + " data: ");
            for (int i = 0; i < k; i++)
            {
                Trace.Write(data[i] + ",");
            }
            Trace.WriteLine("");
        }
        */
        // this bool return true if the plsit makes sense false otherwise
        public unsafe bool GetLeftRightSplit(double* dataPtr, int* idPtr, int dataSize,
            int splitDimension, double splitPoint, out int leftSize)
        {

            int start = 0;
            int end = dataSize - 1;
            double* rt = dataPtr + ((dataSize - 1) * k);
            double* lt = dataPtr;
            int* rtId = idPtr + dataSize - 1;
            int* ltId = idPtr;

            int lftIdx = start;
            int rtIdx = end;

            while (true)
            {
                // run upto the point where lftIdx datum should be in the right side
                while (lftIdx <= end && *(lt + splitDimension) <= splitPoint)
                {
                    lt += k;
                    ltId += 1;
                    lftIdx++;
                }

                while (rtIdx >= start && *(rt + splitDimension) > splitPoint)
                {
                    rt -= k;
                    rtId -= 1;
                    rtIdx--;
                }

                if (lftIdx < rtIdx)
                {
                    //int brId = *rtId;
                    //int blId = *ltId;
                    //double[] ld = { lt[0],lt[1],lt[2],lt[3],lt[4]};
                    //double[] rd = { rt[0],rt[1],rt[2],rt[3],rt[4]};

                    double temp;
                    for (int i = 0; i < k; ++i)
                    {
                        temp = *(rt + i);
                        *(rt + i) = *(lt + i);
                        *(lt + i) = temp;
                    }
                    // also exchange the id's
                    temp = *rtId;
                    *rtId = *ltId;
                    *ltId = (int)temp;

                    /*
                    if (brId != *ltId || blId != *rtId)
                    {
                        Trace.WriteLine("Some Error during exchange");
                        PrintData(ltId, lt);
                        PrintData(rtId, rt);
                    }
                    else
                    {
                        if (lt[0] != rd[0] || lt[1] != rd[1] || lt[2] != rd[2] || lt[3] != rd[3] || lt[4] != rd[4]
                            || rt[0] != ld[0] || rt[1] != ld[1] || rt[2] != ld[2] || rt[3] != ld[3] || rt[4] != ld[4])
                        {
                            Trace.WriteLine("Some Error during exchange");
                            PrintData(ltId, lt);
                            PrintData(rtId, rt);
                        }
                    }
                     */

                }
                else
                {
                    leftSize = rtIdx + 1;
                    break;
                }

            }

            //DateTime ed = DateTime.Now;
            //TimeSpan dur = ed - st;
            //timeInLRS += dur.Ticks;

            if (lftIdx > end || rtIdx < start)
            {
                return false;
            }
            return true;

        }

        public int GetDimensionality()
        {
            return k;
        }

        public INode GetRoot()
        {
            return nodeList[0];
        }

        public int GetThresholdLeafSize()
        {
            return stdlLeafSize;
        }

        public void InitialLizeTree(String dbName, String tableName)
        {
        }

        // returns true if the tree is Uninitialized
        public bool IsEmpty()
        {
            return nodeList[0] == null ? true : false;
        }

        // returns the max number of children pos1 node can have ie fanout
        public int GetFanOut()
        {
            return fanOut;
        }

        public INode[] GetAllNodes()
        {
            return nodeList.ToArray();
        }


        public void SaveToDb(String dataDb, String dataTable, String dbName, String tableName)
        {
            
            Trace.WriteLine("KDTree.SaveToDb()");
            SqlConnection connection = new SqlConnection();
            String mappingTableName = tableName + "_mapping";
            //connection.ConnectionString = "Context Connection=true";
            connection.ConnectionString = "Data Source=KLEENE; Initial Catalog="
                + dbName + "; Integrated Security=True;";
            connection.Open();

            if (MPIRank == 0)
            {
                StoreTreeInformationInMaster(connection, dataDb, dataTable, dbName, tableName, mappingTableName);
                Console.WriteLine("1");
                CreateTreeTable(connection, tableName);
                Console.WriteLine("2");
                CreateNodeDataMappingTable(connection, mappingTableName);
                
            }
            MPIWorld.Barrier();
            SaveNodesToTable(connection, tableName);
            Console.WriteLine("3");
            SaveNodeToDataMapping(connection, mappingTableName);
            Console.WriteLine("4");
            connection.Close();
        }

        private void CreateNodeDataMappingTable(SqlConnection connection, String mappingTableName)
        {
            // create table
            String createTableQuery = TreeUtilities.CREATE_NODE_DATA_MAPPING;
            createTableQuery = createTableQuery.Replace("<MappingTableName>", mappingTableName);
            SqlCommand comm = new SqlCommand();
            comm.Connection = connection;
            comm.CommandText = createTableQuery;
            Console.WriteLine(createTableQuery);
            comm.ExecuteNonQuery();
        }

        private void SaveNodeToDataMapping(SqlConnection connection, String mappingTableName)
        {
            // insert rows
            SqlBulkCopy bulkCopier = new SqlBulkCopy(connection);
            bulkCopier.DestinationTableName = mappingTableName;

            NodeDataMappingReader dataReader =
                new NodeDataMappingReader(nodeDataMapping, ids);
            // free up this memory
            ids = null;
            bulkCopier.WriteToServer(dataReader);
            bulkCopier.Close();
        }

        private void CreateTreeTable(SqlConnection connection, String tableName)
        {
            String createTableQuery = CREATE_TREE_TABLE;
            createTableQuery = createTableQuery.Replace("<TreeTableName>", tableName);
            SqlCommand comm = new SqlCommand();
            comm.Connection = connection;
            comm.CommandText = createTableQuery;
            Console.WriteLine(createTableQuery);
            comm.ExecuteNonQuery();
        }

        private void StoreTreeInformationInMaster(SqlConnection connection,
           String dataDb, String dataTable, String dbName, String tableName, String mappingTableName)
        {
            TreeInfo treeInfo = new TreeInfo(dataDb, dataTable, dbName, tableName, mappingTableName,
                TreeUtilities.KDTreeType, k, fanOut, nNodes, nPoints, GetTreeSpecifics());
            TreeUtilities.WriteTreeInfoToMaster(treeInfo);
        }

        private String GetTreeSpecifics()
        {
            return "[" + maxMedianSplitLevel + "," + stdlLeafSize + "]";
        }

        private void SaveNodesToTable(SqlConnection connection, String tableName)
        {
            
            SqlBulkCopy bulkCopier = new SqlBulkCopy(connection);
            
            bulkCopier.DestinationTableName = tableName;
            
            KDTreeDataReader dataReader =
                new KDTreeDataReader(nodeList, boxMins, boxMaxs, k, nNodes - globalNodeList.Count);
            bulkCopier.BulkCopyTimeout = 90;
            bulkCopier.WriteToServer(dataReader);
            
            dataReader.Close();
            dataReader = new KDTreeDataReader(globalNodeList, globalBoxMins, globalBoxMaxs, k, globalNodeList.Count);
            
            bulkCopier.WriteToServer(dataReader);
            dataReader.Close();
            bulkCopier.Close();

        }

        public void InitializeTree(TreeInfo treeInfo)
        {
            setInternalData(treeInfo);
            KDTreeDataReader.ReadNodes(treeInfo.treeDb, treeInfo.treeTableName, boxMins, boxMaxs, nodeList, k, this);
        }

        private void setInternalData(TreeInfo treeInfo)
        {
            k = treeInfo.dimensionality;
            nNodes = treeInfo.nNodes;
            nPoints = treeInfo.nPoints;
            nodeList = new List<KNode>(treeInfo.nNodes);
            boxMins = new double[k * nNodes];
            boxMaxs = new double[k * nNodes];
            //private int[] nodeDataMapping;          // for data with ids[i],nodeDataMapping[i] is the node id it belongs to 
            //private int* idBegin;                   // the starting address of the ids
            //private int[] ids;                      // the ids of the data, held so that they can be stored.
            //private double* minPtr, maxPtr;         // these are used to iterate through boxMins & boxMaxs
            //private const int maxMedianSplitLevel = -1; /
        }

        private String PrintTime(DateTime startTime, DateTime endTime)
        {
            TimeSpan duration = endTime - startTime;
            /*
            return
                duration.Hours
                + ":" + duration.Minutes
                + ":" + duration.Seconds + "." + duration.Milliseconds;
             */
            return duration.Ticks.ToString();

        }

        public int GetNumNodes()
        {
            return nNodes;
        }

        public int GetDataSize()
        {
            return nPoints;
        }

        public KNode GetNode(int index)
        {
            return nodeList[index];
        }

        public int getNumNodes()
        {
            return nNodes;
        }

        public double GetMinSqEuclideanDst(
                INode node, double[] point)
        {
            KNode nd = (KNode)node;
            double sqrdDistance = 0.0;

            for (int i = 0; i < k; ++i)
            {
                double body_i = point[i];
                double box_i = boxMins[nd.GetBoxIdx() * k + i]; //box.getDimensionMin(i);
                if (body_i < box_i)
                {
                    double delta = (box_i - body_i);
                    sqrdDistance += delta * delta;
                }
                else
                {
                    box_i = boxMaxs[nd.GetBoxIdx() * k + i];//box.getDimensionMax(i);
                    if (body_i > box_i)
                    {
                        double delta = (body_i - box_i);
                        sqrdDistance += delta * delta;
                    }
                }
            }
            return sqrdDistance;

        }

        public double GetMaxSqEuclideanDst(INode node1, INode node2)
        {
            KNode nd1 = (KNode)node1;
            KNode nd2 = (KNode)node2;

            double sqrdDistance = 0.0;

            for (int i = 0; i < k; ++i)
            {
                double box1_i_max = boxMaxs[nd1.GetBoxIdx() * k + i];// box1.getDimensionMax(i);
                double box1_i_min = boxMins[nd1.GetBoxIdx() * k + i];//box1.getDimensionMin(i);
                double box2_i_max = boxMaxs[nd2.GetBoxIdx() * k + i];//box2.getDimensionMax(i);
                double box2_i_min = boxMins[nd2.GetBoxIdx() * k + i];// box2.getDimensionMin(i);

                double max = box1_i_max > box2_i_max ? box1_i_max : box2_i_max;
                double min = box1_i_min < box2_i_min ? box1_i_min : box2_i_min;

                double delta = (max - min);
                sqrdDistance += delta * delta;
            }
            //Trace.WriteLine("Max Sq dist : " + sqrdDistance);
            return sqrdDistance;
        }

        public double GetMinSqEuclideanDst(INode node1, INode node2)
        {
            KNode nd1 = (KNode)node1;
            KNode nd2 = (KNode)node2;
            double sqrdDistance = 0.0;
            for (int i = 0; i < k; ++i)
            {
                double box1_i_max = boxMaxs[nd1.GetBoxIdx() * k + i];// box1.getDimensionMax(i);
                double box1_i_min = boxMins[nd1.GetBoxIdx() * k + i];//box1.getDimensionMin(i);
                double box2_i_max = boxMaxs[nd2.GetBoxIdx() * k + i];//box2.getDimensionMax(i);
                double box2_i_min = boxMins[nd2.GetBoxIdx() * k + i];// box2.getDimensionMin(i);

                if (box1_i_max < box2_i_min)
                {
                    double delta = (box1_i_max - box2_i_min);
                    sqrdDistance += delta * delta;
                }
                else if (box1_i_min > box2_i_max)
                {
                    double delta = (box1_i_min - box2_i_max);
                    sqrdDistance += delta * delta;
                }

            }
            return sqrdDistance;
        }

        unsafe bool DoMedianSplit(double* dataPtr, int* idPtr, int dataSize,
            int splitDimension, out double splitPoint, out int leftSize)
        {
            Quick_sort(0, dataSize - 1, dataPtr, idPtr, splitDimension, (dataSize - 1) / 2);
            int middleIdx = k * (dataSize / 2);
            splitPoint = dataPtr[middleIdx + splitDimension]; // the median
            return GetLeftRightSplit(dataPtr, idPtr, dataSize, splitDimension, splitPoint, out leftSize);
        }

        unsafe void ExchangeData(double* dataPtr, int pos1, int pos2)
        {
            for (int i = 0; i < k; i++)
            {
                double temp = dataPtr[pos1 * k + i];
                dataPtr[pos1 * k + i] = dataPtr[pos2 * k + i];
                dataPtr[pos2 * k + i] = temp;
            }
        }

        unsafe int Partition(int low, int high, double* dataPtr, int* idPtr, int dimension)
        {
            double high_vac, low_vac, pivot;
            int high_vac_id, low_vac_id, pivot_id;
            double* lowPtr = (dataPtr + low * k + dimension);
            double* highPtr = (dataPtr + high * k + dimension);

            /**
             * This is the quick sort partitioning method.
             * The random pivot picked is the first element in the
             * list. The problem is that if we have sorted this dimension before
             * Then this pivot is the lowest element in the list
             * and thus the worst case behaviour of n(n-1)(n-2)... happens.
             * Thus we simply exchange the first element with an element in the middle of the
             * list. 
             */
            int pivot_idx = (high + low) / 2;
            ExchangeData(dataPtr, pivot_idx, low);
            int temp = idPtr[pivot_idx];
            idPtr[pivot_idx] = idPtr[low];
            idPtr[low] = temp;

            // now contnue
            double[] pivotData = new double[k];
            for (int i = 0; i < k; i++)
            {
                pivotData[i] = dataPtr[low * k + i];
            }
            pivot = dataPtr[low * k + dimension];
            pivot_id = idPtr[low];

            while (high > low)
            {
                high_vac = *highPtr; // dataPtr[high * k + dimension];
                high_vac_id = idPtr[high];
                while (pivot < high_vac)
                {
                    if (high <= low)
                    {
                        break;
                    }
                    high--;
                    highPtr -= k;
                    high_vac = *highPtr;
                    high_vac_id = idPtr[high];
                }
                ExchangeData(dataPtr, high, low);   //dataPtr[low * k + dimension] = high_vac;
                idPtr[low] = high_vac_id;
                low_vac = *lowPtr; // dataPtr[low * k + dimension];
                low_vac_id = idPtr[low];

                while (pivot >= low_vac)
                {
                    if (high <= low)
                    {
                        break;
                    }
                    low++;
                    lowPtr += k;
                    low_vac = *lowPtr;
                    low_vac_id = idPtr[low];
                }
                ExchangeData(dataPtr, high, low);
                idPtr[high] = low_vac_id;

            }
            for (int i = 0; i < k; i++)
            {
                dataPtr[low * k + i] = pivotData[i];
            }
            //dataPtr[low * k + dimension] = pivot;
            idPtr[low] = pivot_id;
            return low;
        }

        public unsafe void Quick_sort(int low, int high, double* dataPtr, int* idPtr, int dimension, int middle)
        {

            int Piv_index;
            // Console.WriteLine("Low {0} High {1}", low, high);
            if (low < high)
            {
                Piv_index = Partition(low, high, dataPtr, idPtr, dimension);
                if (Piv_index == middle)
                {
                    return;
                }
                else if (Piv_index > middle)
                {
                    Quick_sort(low, Piv_index - 1, dataPtr, idPtr, dimension, middle);
                }
                else
                {
                    Quick_sort(Piv_index + 1, high, dataPtr, idPtr, dimension, middle);
                }
                //Console.WriteLine("Low {0} Pv {1} High {2}", low, Piv_index, high);
                //Console.ReadLine();
            }
        }

    }



}