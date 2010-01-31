using System;
using System.Diagnostics;
using System.Collections.Generic;
using System.Collections;
using System.Data.SqlTypes;
using Microsoft.SqlServer.Server;
using System.Data.SqlClient;
using System.IO;
using StructureInterfaces;
using Utilities;
using NBodyForMicrosoft2;



namespace Algorithms
{
    unsafe public class DualTreeNN
    {
        // timers
        private long st_SNN;
        private int count_SNN;
        private int count_ignored_leaves;
        private TextWriterTraceListener Tracer;

        private double[] NodeMaxDst;    // for each node in queryTree what is the 
                                        // furthest nearest neighbor found so far

        //Dictionary<int, int> nodeIdIndexMapping;// key= node Id, value= index in double[] NodeMaxDst of req. data

        Dictionary<int, int> qIdIndexMapping;   // key= query point id, value= index in the following 3 data
                                                // data structs. note for nn's and dst's the effective index becomes 
                                                // value * kNN, whereas for maxDist it's value itself
                                                // this is basically because Id's in the target data may be arbitrary

        private int[] nns;          // for each query pt. holds k nn's
        private double[] dsts;      // for each query point holds the distances to the k nn's
        private double[] maxDist;   // for each query point holds the furthest nn till now

        double*[] refNodeIdDataPtr;  // key node id, value pointer to start of data
        double*[] queryNodeIdDataPtr;
        int*[] refNodeIdDataIdPtr;
        int*[] queryNodeIdDataIdPtr;

        private double[] qData;
        private int[] qIds;
        private double[] rData;
        private int[] rIds;

        
        
        
        private List<OutputData> results;
        private bool isSingleTree;  // true if the query and reference trees are the same
            
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

        public void SaveResultsToDb(String dbName, String tableName)
        {
            SqlConnection connection = new SqlConnection();
            //connection.ConnectionString = "Context Connection=true";
            connection.ConnectionString = "Data Source=KLEENE; Initial Catalog=" + dbName + "; Integrated Security=True;";
            connection.Open();
            SqlCommand comm = new SqlCommand(
                    NNResultsReader.GetCreateResultsTableString(tableName), connection);
            comm.ExecuteNonQuery();
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
        private void ReadDataPoints(String treeTableName,
            int*[] nodeIdDataIdPtr, 
            double*[] nodeIdDataPtr, 
            int* idPtr, double* dataPtr, int k)
        {
            SqlConnection connection = new SqlConnection();
            //connection.ConnectionString = "Context Connection=true";
            connection.ConnectionString = "Data Source=KLEENE; Initial Catalog=Master; Integrated Security=True;";
            connection.Open();
                       
            // get the query data
            TreeInfo treeInfo = TreeUtilities.GetTreeInfo(connection, treeTableName);
            String commText = TreeUtilities.GetNodeDataSQLQuery(treeInfo);
            SqlCommand comm = new SqlCommand(commText, connection);
            SqlDataReader rdr = comm.ExecuteReader();
            ParseAndStoreData(rdr, nodeIdDataIdPtr, nodeIdDataPtr, idPtr, dataPtr, k);
            rdr.Close();
            connection.Close();
        }

        private void ParseAndStoreData(SqlDataReader rdr,
            int*[] nodeIdDataIdPtr,
            double*[] nodeIdDataPtr, 
            int* ids, double* data, int k)
        {
            int prevNodeId = -1;
            int nodeId;
            
            if( rdr.Read() )
            {
                nodeId = rdr.GetInt32(0);
                prevNodeId = nodeId;

                nodeIdDataPtr[nodeId - 1] = data;
                nodeIdDataIdPtr[nodeId - 1] = ids;
                
                *ids = rdr.GetInt32(1);
                ids++;
                for (int i = 2; i < k + 2; ++i)
                {
                    *data = rdr.GetDouble(i);
                    data++;
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
                    nodeIdDataPtr[nodeId - 1] = data;
                    nodeIdDataIdPtr[nodeId - 1] = ids;
                }

                *ids = rdr.GetInt32(1);
                ids++;

                for (int i = 2; i < k + 2; ++i)
                {
                    *data = rdr.GetDouble(i);
                    data++;
                }
                prevNodeId = nodeId;
            }
        }

        
        
        /// <summary>
        /// Given the Query tree which is the tree for which results are to be returned
        /// various data structures need to be setup for each query point, query node etc.
        /// This function handles that before the algorithm actually begins.
        /// </summary>
        private void InitQueryPointDataStructures(ITree queryTree, int* qIds, int kNN)
        {
            double max = double.MaxValue;
            for (int i = 0; i < kNN * queryTree.GetDataSize(); ++i){
                dsts[i] = max;
            }

            for (int i = 0; i < queryTree.GetDataSize(); ++i){
                maxDist[i] = max;
                qIdIndexMapping.Add(*qIds, i);
                qIds++;
            }
            INode[] nodes = queryTree.GetAllNodes();
            for (int i = 0; i < queryTree.GetNumNodes(); ++i)
            {
                NodeMaxDst[i] = max;
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
            isSingleTree = true;
            ITree tree = TreeUtilities.GetTreeFromTableName(treeTableName);

            // allocate the data structures and then fix them
            NodeMaxDst = new double[tree.GetNumNodes()];
            //nodeIdIndexMapping = new Dictionary<int, int>(queryTree.GetNumNodes());
            qIdIndexMapping = new Dictionary<int, int>(tree.GetDataSize());
            nns = new int[kNN * tree.GetDataSize()];
            dsts = new double[kNN * tree.GetDataSize()];
            maxDist = new double[tree.GetDataSize()];

            queryNodeIdDataPtr = new double*[tree.GetNumNodes()];
            queryNodeIdDataIdPtr = new int*[tree.GetNumNodes()];
            qData = new double[tree.GetDataSize() * tree.GetDimensionality()];
            qIds = new int[tree.GetDataSize()];

            refNodeIdDataPtr = queryNodeIdDataPtr;
            refNodeIdDataIdPtr = queryNodeIdDataIdPtr;
            rData = qData;
            rIds = qIds;

            IEnumerable results;
            System.GC.Collect();
            // now fix the data and run 
            fixed( double* qDataPtr = qData, rDataPtr = rData )
            {
                fixed( int* qIdPtr = qIds, rIdPtr = rIds )
                {
                    ReadDataPoints(treeTableName, queryNodeIdDataIdPtr, 
                        queryNodeIdDataPtr, qIdPtr, qDataPtr, tree.GetDimensionality());
                    InitQueryPointDataStructures(tree, qIdPtr, kNN);
                    results = DualTreeNearestNeighbors(tree, tree, kNN);
                }
            }
            
            CloseEnvironment();
            return results;
        }


        [SqlFunction(Name = "NN",
        DataAccess = DataAccessKind.Read,
        IsDeterministic = true,
        IsPrecise = true,
        TableDefinition = "QueryPointId int, ReferencePointId int, Distance double",
        FillRowMethodName = "OutputContract")]
        public IEnumerable GetNearestNeighbors(
            String qTreeTableName, String rTreeTableName, int kNN)
        {
            if( qTreeTableName.ToLower().Equals(qTreeTableName.ToLower()) )
            {
                throw new InvalidOperationException();
            }

            isSingleTree = false;
            SetupEnvironment();
            ITree queryTree = TreeUtilities.GetTreeFromTableName(qTreeTableName);
            ITree refTree = TreeUtilities.GetTreeFromTableName(rTreeTableName);

            // allocate the data structures and then fix them
            NodeMaxDst = new double[queryTree.GetNumNodes()];
            //nodeIdIndexMapping = new Dictionary<int, int>(queryTree.GetNumNodes());
            qIdIndexMapping = new Dictionary<int, int>(queryTree.GetDataSize());
            nns = new int[kNN * queryTree.GetDataSize()];
            dsts = new double[kNN * queryTree.GetDataSize()];
            maxDist = new double[queryTree.GetDataSize()];

            queryNodeIdDataPtr = new double*[queryTree.GetNumNodes()];
            queryNodeIdDataIdPtr = new int*[queryTree.GetNumNodes()];
            qData = new double[queryTree.GetDataSize() * queryTree.GetDimensionality()];
            qIds = new int[queryTree.GetDataSize()];

            refNodeIdDataPtr = new double*[refTree.GetNumNodes()];
            refNodeIdDataIdPtr = new int*[refTree.GetNumNodes()];
            rData = new double[refTree.GetDataSize() * refTree.GetDimensionality()];
            rIds = new int[refTree.GetDataSize()];

            IEnumerable results;
            fixed (double* qDataPtr = qData, rDataPtr = rData)
            {
                fixed (int* qIdPtr = qIds, rIdPtr = rIds)
                {
                    ReadDataPoints(qTreeTableName, queryNodeIdDataIdPtr,
                        queryNodeIdDataPtr, qIdPtr, qDataPtr, queryTree.GetDimensionality());
                    ReadDataPoints(rTreeTableName, refNodeIdDataIdPtr,
                        refNodeIdDataPtr, rIdPtr, rDataPtr, refTree.GetDimensionality());

                    InitQueryPointDataStructures(queryTree, qIdPtr, kNN);
                    results = DualTreeNearestNeighbors(queryTree, refTree, kNN);
                }
            }

            CloseEnvironment();
            return results;
        }

        private IEnumerable DualTreeNearestNeighbors(
            ITree queryTree, ITree refTree, int kNN)
        {
            st_SNN = 0;
            count_SNN = 0;
            count_ignored_leaves = 0;
            double queryRefMinSquaredDistance =
                queryTree.GetMinSqEuclideanDst(refTree.GetRoot(), queryTree.GetRoot());
            DateTime startTime, endTime;
            TimeSpan duration;
            startTime = DateTime.Now;
                
            RecursiveDualTree(queryTree.GetRoot(), refTree.GetRoot(),
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

            results = new List<OutputData>();
            IEnumerator<int> it = qIdIndexMapping.Keys.GetEnumerator();
            
            while (it.MoveNext())
            {
                
                int qId = it.Current;
                int qIndex;
                if( ! qIdIndexMapping.TryGetValue(qId, out qIndex) )
                    Console.WriteLine("shout9");
                //Trace.Write(qIndex + ", ");
                for (int i = 0; i < kNN; i++)
                {
                    OutputData output = new OutputData();
                    output.queryPointId = qId;
                    output.referencePointId = nns[qIndex * kNN + i];
                    output.distance = dsts[qIndex * kNN + i];
                    //TraceThisOP(output);
                    results.Add(output);
                }
            }
            return results;

        }

        private void TraceThisOP(OutputData o)
        {
            Trace.WriteLine(o.queryPointId + "\t\t" + o.referencePointId + "\t\t" + o.distance);
        }

        private void RecursiveDualTree(INode queryNode, INode refNode,
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
                    INode LC = queryNode.GetChild(0);
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

                    INode RC = queryNode.GetChild(1);
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

        private void GetQueryPoints(INode qNode, out double* points, out int* ids)
        {
            int nodeId = qNode.GetNodeId();
            points = queryNodeIdDataPtr[nodeId - 1];
            ids = queryNodeIdDataIdPtr[nodeId - 1];
        }

        private void GetReferencePoints(INode rNode, out double* points, out int* ids)
        {
            int nodeId = rNode.GetNodeId();
            points = queryNodeIdDataPtr[nodeId - 1];
            ids = queryNodeIdDataIdPtr[nodeId - 1];
        }

        private double GetSqDistance(double* q, int i, double* r, int j, int k)
        {
            double dst = 0;
            int iOffset = i * k;
            int jOffset = j * k;
            for (int p = 0; p < k; ++p)
            {
                double delta = (q[iOffset++] - r[jOffset++]) ;
                dst += delta * delta;
            }
            count_SNN++;
            return dst;
        }

        private void SetNearestNeigborsForLeafs(
            INode queryNode, INode refNode, int kNN)
        {
            int k = queryNode.GetTree().GetDimensionality();    // dimensionality
            int I = queryNode.GetNumPoints();   // number of query points
            int J = refNode.GetNumPoints();     // number of reference points
            
            // get the query and reference data
            double* queryPoints, refPoints;
            int* queryIds, refIds;
            GetQueryPoints(queryNode, out queryPoints, out queryIds);
            GetReferencePoints(refNode, out refPoints, out refIds);
            

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
                    refNode, queryPoints + (i * k) );
                
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
                    if ( isSingleTree && queryIds[i] == refIds[j])
                    {
                        // we dont worry about maxOfMax here
                        continue;
                    }

                    double distance = GetSqDistance(queryPoints, i, refPoints, j, k);
                    // nns dsts maxDst
                    if (distance < maxDistance)
                    {
                        // this distance should go into the nn results for this query
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
        /// This function puts the rId in the nearest neighbor results and updates the nns, dsts and maxDst
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
            
            for(int i = 0; i < kNN; ++i)
            {
                if (dsts[qIndex*kNN + i] == maxDistance)
                {
                    maxIdx = i;
                    for ( int j = i + 1; j < kNN; ++j)
                    {
                        if (secondLargest < dsts[qIndex * kNN + j])
                        {
                            secondLargest = dsts[qIndex * kNN + j];
                        }
                    }
                    break;
                }
                else if( secondLargest < dsts[qIndex*kNN + i] )
                {
                    secondLargest = dsts[qIndex*kNN + i];
                    

                }
            }
            // update nns etc
            nns[qIndex*kNN + maxIdx] = rId;
            dsts[qIndex*kNN + maxIdx] = distance;

            // check if second largest idx was updated. it will only not
            // get updated if the whole array of dsts contains only max distances
            // for this queryid. In this case maxIdx is gaurunteed to end up as kNN-1
            // and thus setting the second largest ids and value to 0 index does the trick
            if (secondLargest == -1) 
            { 
                // it wasn't. set the value to 
                secondLargest = dsts[qIndex * kNN];
            }
    
            maxDist[qIndex] = secondLargest > distance? secondLargest : distance;
            return maxDist[qIndex];

        }
        /// <summary>
        /// Gets the value for the furthest NN found for the points under this
        /// query node.
        /// </summary>
        /// <param name="node"></param>
        /// <returns></returns>
        private double GetQNodePruneDistance(INode queryNode)
        {
            int index = queryNode.GetNodeId() - 1;
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
        private void SetQNodePruneDistance(INode queryNode, double pruneDistance)
        {
            int index = queryNode.GetNodeId() - 1;
            //if( !nodeIdIndexMapping.TryGetValue(queryNode.GetNodeId(), out index) )
            //    Console.WriteLine("shout6");
            if (NodeMaxDst[index] < pruneDistance)
            {
                throw new Exception( "Changing prune dist from " + NodeMaxDst[index] + " to " + pruneDistance + " for node " + queryNode.GetNodeId());
            }
            NodeMaxDst[index] = pruneDistance;
        }
    }
}