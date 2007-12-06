using System;
using System.Diagnostics;
using System.Collections.Generic;
using System.Collections;
using System.Data.SqlTypes;
using Microsoft.SqlServer.Server;
using System.Data.SqlClient;
using System.IO;
using Utilities;
using KDTreeStructures;
using MPI;



namespace Algorithms
{
    public class DualTreeNN
    {
        // timers
        private long st_SNN;
        private int count_SNN;
        private int count_ignored_leaves;
        private TextWriterTraceListener Tracer;

        private double[] NodeMaxDst;    // for each node in tree what is the 
        // furthest nearest neighbor found so far

        //Dictionary<int, int> nodeIdIndexMapping;// key= node Id, value= index in double[] NodeMaxDst of req. data

        Dictionary<int, int> qIdIndexMapping;   // key= query point id, value= index in the following 3 data
        // data structs. note for nn's and dst's the effective index becomes 
        // value * kNN, whereas for maxDist it's value itself
        // this is basically because Id's in the target data may be arbitrary

        private int[] nns;          // for each query pt. holds k nn's
        private double[] dsts;      // for each query point holds the distances to the k nn's
        private double[] maxDist;   // for each query point holds the furthest nn till now

        private Dictionary<int, double[]> data;
        private Dictionary<int, int[]> dataIds;


        // memebers to help in parallelization
        KNode subTreeRoot;
        int subTreeNumNodes;

        List<OutputData> results;

        /**
         * Following are the MPI variables
         */
        private int MPIRank;
        private int MPIWorldSize;
        private Communicator MPIWorld;

        /// <summary>
        /// Initializes all the MPI variables that are required by the program.
        /// </summary>
        private void InitMPIVars()
        {
            MPIWorld = Communicator.world;
            MPIRank = Communicator.world.Rank;
            MPIWorldSize = Communicator.world.Size;
        }

        public struct OutputData
        {
            public int queryPointId;
            public int referencePointId;
            public double distance;

            public OutputData(int qId, int rId, double dst)
            {
                queryPointId = qId;
                referencePointId = rId;
                distance = dst;
            }
        }


        public void OutputContract(object output, out SqlInt32 queryPointId,
            out SqlInt32 referencePointId, out SqlDouble distance)
        {
            OutputData outputData = (OutputData)output;
            queryPointId = outputData.queryPointId;
            referencePointId = outputData.referencePointId;
            distance = outputData.distance;
        }

        private void SetupTracing()
        {
            if (Tracer == null)
            {
                Tracer = new
                 TextWriterTraceListener("C:\\Documents and Settings\\manyu\\Desktop\\trace.txt");
                Trace.Listeners.Add(Tracer);
            }

            Trace.WriteLine("Starting Trace.");
            Trace.Flush();
        }

        private void SetupEnvironment()
        {
            SetupTracing();
        }

        private void CloseEnvironment()
        {
            Trace.Flush();
            Trace.Close();
        }

        /// <summary>
        /// This function reads the actual data from the tables. If isSingleTree is
        /// true is does not read the data twice but makes the query and reference 
        /// pointers point to the same structures.
        /// </summary>
        private void ReadDataPoints(String qTreeTableName, int k)
        {
            SqlConnection connection = new SqlConnection();
            //connection.ConnectionString = "Context Connection=true";
            connection.ConnectionString = "Data Source=KLEENE; Initial Catalog=Master; Integrated Security=True;";
            connection.Open();

            // get the query data
            TreeInfo qTreeInfo = TreeUtilities.GetTreeInfo(connection, qTreeTableName);
            String commText = TreeUtilities.GetNodeDataSQLQuery(qTreeInfo);
            SqlCommand comm = new SqlCommand(commText, connection);
            SqlDataReader rdr = comm.ExecuteReader();
            ParseAndStoreData(rdr, dataIds, data, k);
            rdr.Close();
            connection.Close();
        }

        private void ParseAndStoreData(SqlDataReader rdr,
            Dictionary<int, int[]> ids, Dictionary<int, double[]> data, int k)
        {
            int prevNodeId = -1;
            int nodeId;
            List<double> doubleList = new List<double>(50 * k);
            List<int> idList = new List<int>(50);

            if (rdr.Read())
            {
                nodeId = rdr.GetInt32(0);
                prevNodeId = nodeId;
                idList.Add(rdr.GetInt32(1));
                for (int i = 2; i < k + 2; ++i)
                {
                    doubleList.Add(rdr.GetDouble(i));
                }
            }
            else
            {
                throw new Exception();
            }


            while (rdr.Read())
            {
                nodeId = rdr.GetInt32(0);
                if (nodeId != prevNodeId)
                {
                    // read the data and add to current data structures
                    data.Add(prevNodeId, doubleList.ToArray());
                    ids.Add(prevNodeId, idList.ToArray());
                    doubleList = new List<double>(50 * k);
                    idList = new List<int>(50);
                }
                idList.Add(rdr.GetInt32(1));
                for (int i = 2; i < k + 2; ++i)
                {
                    doubleList.Add(rdr.GetDouble(i));
                }
                prevNodeId = nodeId;
            }
            data.Add(prevNodeId, doubleList.ToArray());
            ids.Add(prevNodeId, idList.ToArray());

        }

        private void Print(String str)
        {
            Console.WriteLine("Rank " + MPIRank + ": " + str);
        }
        private void FindQuerySubTree(KDTree queryTree)
        {
            
            subTreeRoot = queryTree.GetRoot();
            
            int seperateLevel = (int)Math.Log(MPIWorldSize, 2);
            Print("" + seperateLevel);
            for (int i = 0; i < seperateLevel; i++)
            {
                if( subTreeRoot.IsLeaf() )
                {
                    throw new Exception("The query tree ends even before we can split the work");
                }
                int rankShifter = MPIRank;
                rankShifter = rankShifter >> seperateLevel - i - 1;
                if (rankShifter % 2 == 1)   // go to the right sub part
                {
                    Print("Moving Right");
                    subTreeRoot = subTreeRoot.GetChild(1);
                }
                else
                {
                    // keep to the left sub part
                    Print("Moving Left");
                    subTreeRoot = subTreeRoot.GetChild(0);
                }
            }
            // now find the right most leaf of the subtree
            KNode rightmost = subTreeRoot;
            while(!rightmost.IsLeaf())
            {
                rightmost = rightmost.GetChild(1);
            }
            subTreeNumNodes = rightmost.GetNodeId() - subTreeRoot.GetNodeId() + 1;
            
        }

        /// <summary>
        /// Given the Query tree which is the tree for which results are to be returned
        /// various data structures need to be setup for each query point, query node etc.
        /// This function handles that before the algorithm actually begins.
        /// </summary>
        private void InitDataStructures(KDTree tree, String treeTableName, int kNN)
        {
            FindQuerySubTree(tree);

            NodeMaxDst = new double[subTreeNumNodes];
            qIdIndexMapping = new Dictionary<int, int>(subTreeRoot.GetNumPoints());
            
            nns = new int[kNN * subTreeRoot.GetNumPoints()];
            dsts = new double[kNN * subTreeRoot.GetNumPoints()];
            maxDist = new double[subTreeRoot.GetNumPoints()];

            data = new Dictionary<int, double[]>(tree.GetNumNodes());
            dataIds = new Dictionary<int, int[]>(tree.GetNumNodes());

            
            ReadDataPoints(treeTableName, tree.GetDimensionality());
            double max = double.MaxValue;
            for (int i = 0; i < kNN * subTreeRoot.GetNumPoints(); ++i)
            {
                dsts[i] = max;
            }
            
            for (int i = 0; i < subTreeRoot.GetNumPoints(); ++i)
            {
                maxDist[i] = max;
            }
            KNode[] nodes = tree.GetAllNodes();
            int count = 0;
            
            int subTreeRootNodeId = subTreeRoot.GetNodeId();
            for (int i = 0; i < subTreeNumNodes; ++i)
            {
                NodeMaxDst[i] = max;
                // now update the queryPt index mapping data structure
                int[] qPointIds;
                
                if (dataIds.TryGetValue(nodes[subTreeRootNodeId - 1 + i].GetNodeId(), out qPointIds))
                {

                    for (int j = 0; j < qPointIds.Length; j++)
                    {
                        qIdIndexMapping.Add(qPointIds[j], count);
                        count++;
                    }
                }
                else if( nodes[subTreeRootNodeId - 1 + i].IsLeaf() )
                {
                    throw new Exception("System is inc consistant");
                }
            }
            
        }

        [SqlFunction(Name = "NN",
        DataAccess = DataAccessKind.Read,
        IsDeterministic = true,
        IsPrecise = true,
        TableDefinition = "QueryPointId int, ReferencePointId int, Distance double",
        FillRowMethodName = "OutputContract")]
        public IEnumerable GetNearestNeighbors(String treeTableName, int kNN)
        {
            SetupEnvironment();
            Trace.WriteLine("Read data from tables");
            Trace.Flush();
            KDTree tree = TreeUtilities.GetTreeFromTableName(treeTableName);
            double startTime, endTime;
            startTime = MPI.Environment.Wtime;
            InitMPIVars();
            InitDataStructures(tree, treeTableName, kNN);
            IEnumerable results = DualTreeNearestNeighbors(tree, kNN);
            endTime = MPI.Environment.Wtime;
            double t1 = endTime - startTime;
            Console.WriteLine("Rank " + MPI.Communicator.world.Rank
                    + "Time taken = " + (t1));
            Trace.Flush();
            CloseEnvironment();
            return results;
        }

        private IEnumerable DualTreeNearestNeighbors(
            KDTree tree, int kNN)
        {
            st_SNN = 0;
            count_SNN = 0;
            count_ignored_leaves = 0;

            double queryRefMinSquaredDistance =
                tree.GetMinSqEuclideanDst(subTreeRoot, tree.GetRoot());
            
            DateTime startTime, endTime;
            TimeSpan duration;
            startTime = DateTime.Now;

            RecursiveDualTree(subTreeRoot, tree.GetRoot(),
                queryRefMinSquaredDistance, kNN);

            endTime = DateTime.Now;
            duration = endTime - startTime;
            Trace.WriteLine("Recusrive Dual Tree took = " +
                duration.Minutes + "m, " + duration.Seconds + "." + duration.Milliseconds + "s");
            Trace.WriteLine("Recursive dual tree took : " + duration.Ticks);
            Trace.WriteLine("Set NN for leafs took : " + st_SNN);
            Trace.WriteLine("Number of leaves compared is : " + count_SNN);
            Trace.WriteLine("Count NN for leafs ignored : " + count_ignored_leaves);

            //Trace.WriteLine("Hash Table Size: " + hashTable.Count);
            //Trace.WriteLine("table size : " + table.Count);
            Trace.Flush();

            IEnumerator<int> it = qIdIndexMapping.Keys.GetEnumerator();
            results = new List<OutputData>(tree.GetDataSize() * kNN);
            while (it.MoveNext())
            {

                int qId = it.Current;
                int qIndex;
                if (!qIdIndexMapping.TryGetValue(qId, out qIndex))
                    throw new Exception("The system is inconsistant");
                //Trace.Write(qIndex + ", ");
                for (int i = 0; i < kNN; i++)
                {
                    OutputData output = new OutputData();
                    output.queryPointId = qId;
                    output.referencePointId = nns[qIndex * kNN + i];
                    output.distance = dsts[qIndex * kNN + i];
                    //if( qId > 100 && qId < 200 )
                        //TraceThisOP(output);
                    results.Add(output);
                }
            }
            return results;

        }

        public void SaveResultsToDb(String dbName, String tableName)
        {
            SqlConnection connection = new SqlConnection();
            //connection.ConnectionString = "Context Connection=true";
            connection.ConnectionString = "Data Source=KLEENE; Initial Catalog=" + dbName + "; Integrated Security=True;";
            connection.Open();
            if (MPIRank == 0)  // create the table
            {
                SqlCommand comm = new SqlCommand(
                    NNResultsReader.GetCreateResultsTableString(tableName), connection);
                comm.ExecuteNonQuery();
            }
            MPIWorld.Barrier(); // every one waits till the table is created
            SqlBulkCopy bulkCopier = new SqlBulkCopy(connection);
            bulkCopier.DestinationTableName = tableName;

            NNResultsReader dataReader =
                new NNResultsReader(results);
            //bulkCopier.BulkCopyTimeout = 90;
            bulkCopier.WriteToServer(dataReader);
            dataReader.Close();
            bulkCopier.Close();
            connection.Close();
        }

        private void TraceThisOP(OutputData o)
        {
            Trace.WriteLine(o.queryPointId + "\t\t" + o.referencePointId + "\t\t" + o.distance);
        }

        private void RecursiveDualTree(KNode queryNode, KNode refNode,
            double queryRefMinSqDst, int kNN)
        {
            if (GetQNodePruneDistance(queryNode) < queryRefMinSqDst)
            {
            }
            else
            {
                //CompareAndUpdateQueryAndReferenceNodes(queryNode, refNode);
                // ok now we must check all 4 conditions
                if (queryNode.IsLeaf() && refNode.IsLeaf())
                {
                    SetNearestNeigborsForLeafs(queryNode, refNode, kNN);

                }
                else if (!queryNode.IsLeaf() && !refNode.IsLeaf())
                {
                    // first try and prune one half of the query tree just based on dst
                    // from this query root before going to its children
                    // examine left child and then right child
                    KNode LC = queryNode.GetChild(0);
                    double LCLCdist = queryNode.GetTree().GetMinSqEuclideanDst(LC, refNode.GetChild(0));
                    double LCRCdist = queryNode.GetTree().GetMinSqEuclideanDst(LC, refNode.GetChild(1));

                    if (LCLCdist < LCRCdist)
                    {
                        RecursiveDualTree(LC, refNode.GetChild(0), LCLCdist, kNN);
                        RecursiveDualTree(LC, refNode.GetChild(1), LCRCdist, kNN);
                    }
                    else
                    {
                        RecursiveDualTree(LC, refNode.GetChild(1), LCRCdist, kNN);
                        RecursiveDualTree(LC, refNode.GetChild(0), LCLCdist, kNN);
                    }

                    KNode RC = queryNode.GetChild(1);
                    double RCLCdist = queryNode.GetTree().GetMinSqEuclideanDst(RC, refNode.GetChild(0));
                    double RCRCdist = queryNode.GetTree().GetMinSqEuclideanDst(RC, refNode.GetChild(1));
                    if (RCLCdist < RCRCdist)
                    {
                        RecursiveDualTree(RC, refNode.GetChild(0), RCLCdist, kNN);
                        RecursiveDualTree(RC, refNode.GetChild(1), RCRCdist, kNN);
                    }
                    else
                    {
                        RecursiveDualTree(RC, refNode.GetChild(1), RCRCdist, kNN);
                        RecursiveDualTree(RC, refNode.GetChild(0), RCLCdist, kNN);
                    }
                    double lcPrune = GetQNodePruneDistance(LC);
                    double rcPrune = GetQNodePruneDistance(RC);
                    Double maxPruneDst = lcPrune > rcPrune ? lcPrune : rcPrune;

                    SetQNodePruneDistance(queryNode, maxPruneDst);
                }
                else if (queryNode.IsLeaf() && !refNode.IsLeaf())
                {
                    //Trace.WriteLine("Case2");
                    double LCdist = queryNode.GetTree().GetMinSqEuclideanDst(queryNode, refNode.GetChild(0));
                    double RCdist = queryNode.GetTree().GetMinSqEuclideanDst(queryNode, refNode.GetChild(1));
                    if (LCdist < RCdist)
                    {
                        RecursiveDualTree(queryNode, refNode.GetChild(0), LCdist, kNN);
                        RecursiveDualTree(queryNode, refNode.GetChild(1), RCdist, kNN);
                    }
                    else
                    {
                        RecursiveDualTree(queryNode, refNode.GetChild(1), RCdist, kNN);
                        RecursiveDualTree(queryNode, refNode.GetChild(0), LCdist, kNN);
                    }
                }
                else if (!queryNode.IsLeaf() && refNode.IsLeaf())
                {
                    double LCdist = queryNode.GetTree().GetMinSqEuclideanDst(queryNode.GetChild(0), refNode);
                    double RCdist = queryNode.GetTree().GetMinSqEuclideanDst(queryNode.GetChild(1), refNode);
                    RecursiveDualTree(queryNode.GetChild(0), refNode, LCdist, kNN);
                    RecursiveDualTree(queryNode.GetChild(1), refNode, RCdist, kNN);
                    double lcPrune = GetQNodePruneDistance(queryNode.GetChild(0));
                    double rcPrune = GetQNodePruneDistance(queryNode.GetChild(1));
                    Double maxPruneDst = lcPrune > rcPrune ? lcPrune : rcPrune;
                    SetQNodePruneDistance(queryNode, maxPruneDst);
                }
            }
        }

        private void GetPoints(KNode node, out double[] points, out int[] ids)
        {
            //Trace.WriteLine(qNode.GetNodeId());
            if (!data.TryGetValue(node.GetNodeId(), out points))
            {
                throw new Exception("The system is inconsistant");
            }
            if (!dataIds.TryGetValue(node.GetNodeId(), out ids))
            {
                throw new Exception("The system is inconsistant");
            }
        }

        private double GetSqDistance(double[] q, int i, double[] r, int j, int k)
        {
            double dst = 0;
            int iOffset = i * k;
            int jOffset = j * k;
            for (int p = 0; p < k; ++p)
            {
                double delta = (q[iOffset++] - r[jOffset++]);
                dst += delta * delta;
            }
            count_SNN++;
            return dst;
        }

        private double[] GetArraySegment(double[] array, int start, int size)
        {
            double[] ret = new double[size];
            Array.Copy(array, start, ret, 0, size);
            return ret;
        }

        private void SetNearestNeigborsForLeafs(
            KNode queryNode, KNode refNode, int kNN)
        {
            int k = queryNode.GetTree().GetDimensionality();    // dimensionality
            int I = queryNode.GetNumPoints();   // number of query points
            int J = refNode.GetNumPoints();     // number of reference points

            // get the query and reference data
            double[] queryPoints, refPoints;
            int[] queryIds, refIds;
            GetPoints(queryNode, out queryPoints, out queryIds);
            GetPoints(refNode, out refPoints, out refIds);

            double maxOfMax = 0;
            for (int i = 0; i < I; ++i)     // for each query point
            {
                // get the index of the query point
                int queryPointIdx;
                if (!qIdIndexMapping.TryGetValue(queryIds[i], out queryPointIdx))
                {
                    throw new Exception("The system is inconsistant");
                }
                double maxDistance = maxDist[queryPointIdx];

                // first prune based on the distance from this point to the Ref Node
                double qPointRefNodeDst = queryNode.GetTree().GetMinSqEuclideanDst(
                    refNode, GetArraySegment(queryPoints, i * k, k));

                // prune of this qPoint cannot contain a NN in this ref node
                if (qPointRefNodeDst > maxDistance)
                {
                    if (maxOfMax < maxDistance)
                    {
                        maxOfMax = maxDistance;
                    }
                    continue;
                }

                for (int j = 0; j < J; ++j)
                {
                    // check if it is a single tree and the id's are the same
                    if (queryIds[i] == refIds[j])
                    {
                        // we dont worry about maxOfMax here
                        continue;
                    }

                    double distance = GetSqDistance(queryPoints, i, refPoints, j, k);
                    // nns dsts maxDst
                    if (distance < maxDistance)
                    {
                        // this distance should go into the nn list for this query
                        // put this one in the max's position and change maxDst and pruneDistance
                        maxDistance = AddNearestNeighbor(queryIds[i], refIds[j], distance, maxDistance, kNN);
                    }
                }

                if (maxOfMax < maxDistance)
                {
                    maxOfMax = maxDistance;
                }
            }
            //Console.WriteLine("flags 1");
            SetQNodePruneDistance(queryNode, maxOfMax);

        }

        /// <summary>
        /// This function puts the rId in the nearest neighbor list and updates the nns, dsts and maxDst
        /// data structures. It puts the rId in the place of the furthest NN to the query point. Then
        /// it returns the new furthest NN.
        /// </summary>
        /// <returns>The new furthest NN for this qId.</returns>
        private double AddNearestNeighbor(int qId, int rId, double distance, double maxDistance, int kNN)
        {
            double secondLargest = -1;
            int maxIdx = -1;
            int qIndex;
            if (!qIdIndexMapping.TryGetValue(qId, out qIndex))
            {
                throw new Exception("The system is inconsistant");
            }

            for (int i = 0; i < kNN; ++i)
            {
                if (dsts[qIndex * kNN + i] == maxDistance)
                {
                    maxIdx = i;
                    for (int j = i + 1; j < kNN; ++j)
                    {
                        if (secondLargest < dsts[qIndex * kNN + j])
                        {
                            secondLargest = dsts[qIndex * kNN + j];
                        }
                    }
                    break;
                }
                else if (secondLargest < dsts[qIndex * kNN + i])
                {
                    secondLargest = dsts[qIndex * kNN + i];


                }
            }
            // update nns etc
            nns[qIndex * kNN + maxIdx] = rId;
            dsts[qIndex * kNN + maxIdx] = distance;

            // check if second largest idx was updated. it will only not
            // get updated if the whole array of dsts contains only max distances
            // for this queryid. In this case maxIdx is gaurunteed to end up as kNN-1
            // and thus setting the second largest ids and value to 0 index does the trick
            if (secondLargest == -1)
            {
                // it wasn't. set the value to 
                secondLargest = dsts[qIndex * kNN];
            }

            maxDist[qIndex] = secondLargest > distance ? secondLargest : distance;
            return maxDist[qIndex];

        }
        /// <summary>
        /// Gets the value for the furthest NN found for the points under this
        /// query node.
        /// </summary>
        /// <param name="node"></param>
        /// <returns></returns>
        private double GetQNodePruneDistance(KNode queryNode)
        {
            int index = queryNode.GetNodeId() - subTreeRoot.GetNodeId();
            //if( !nodeIdIndexMapping.TryGetValue(queryNode.GetNodeId(), out index) )
            //    Console.WriteLine("shout7");
            return NodeMaxDst[index];
        }

        /// <summary>
        /// Sets the value for the furthest NN found for the points under this
        /// query node.
        /// </summary>
        /// <param name="queryNode"></param>
        /// <param name="pruneDistance"></param>
        private void SetQNodePruneDistance(KNode queryNode, double pruneDistance)
        {
            int index = queryNode.GetNodeId() - subTreeRoot.GetNodeId();
            //if( !nodeIdIndexMapping.TryGetValue(queryNode.GetNodeId(), out index) )
            //    Console.WriteLine("shout6");
            if (NodeMaxDst[index] < pruneDistance)
            {
                throw new Exception("Changing prune dist from " + NodeMaxDst[index] + " to " + pruneDistance + " for node " + queryNode.GetNodeId());
            }
            NodeMaxDst[index] = pruneDistance;
        }
    }
}