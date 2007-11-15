using System;
using System.Collections.Generic;
using System.Text;
using System.Runtime.InteropServices;
using StructureInterfaces;
using System.Diagnostics;
[assembly: CLSCompliant(true)]

namespace KDTreeStructures
{
    /// <summary>
    /// This is the KD Tree class.
    /// </summary>
    public sealed unsafe class KDTree : ITree
    {
        /**
         * Debugging members.
         */
        private static TextWriterTraceListener Tracer;
        private DateTime startTime = DateTime.Now;
        private DateTime endTime = DateTime.Now;

        /**
         * Data Members
         */
        private int k;
        private const int idealLeafSize = 30;
        private const int fanOut = 2;
        private int nNodes;
        private List<KNode> nodeList;
        private double[] boxMins;
        private double[] boxMaxs;
        private double* minPtr, maxPtr;

        public KDTree()
        {
            // junk delete this
            Tracer = new
               TextWriterTraceListener("C:\\Documents and Settings\\manyu\\Desktop\\trace.txt");
            Trace.Listeners.Add(Tracer);
        }
                

        /// <summary>
        /// This method creates the tree from the input data. 
        /// The assumption here is that the data is normalized. 
        /// Currently we do not worry about overflowing the initial size 
        /// but that has to be taken care of. 
        /// </summary>
        /// <param name="data">The data used to create the tree</param>
        public unsafe void InitializeTree(int[] ids, double[] data)
        {
            k = data.Length/ids.Length;
            Trace.WriteLine("KDTree.InitializeTree(Datum[]): Set the tree dimensionality to " + k);
            int initialArraySize = GetInitialArraySize(ids.Length);
            Trace.WriteLine("KDTree.InitializeTree(Datum[]): DataLength is "
                + ids.Length + " and initial node array size set to "
                + initialArraySize);

            System.GC.Collect();
            nodeList = new List<KNode>(initialArraySize);
            boxMins = new double[initialArraySize*k];
            boxMaxs = new double[initialArraySize*k];
            
            fixed (double* dataP = data, minP = boxMins, maxP = boxMaxs)
            {
                fixed (int* idP = ids)
                {
                    minPtr = minP; 
                    maxPtr = maxP;
                    nNodes = 0;
                    Trace.WriteLine("KDTree.InitializeTree(Datum[]): Calling CreateTreeRecursively().");
                    /**
                     * Create tree recursively
                     */
                    startTime = DateTime.Now;
                    CreateTreeRecursively(dataP, idP, ids.Length, 0);
                    endTime = DateTime.Now;
                }
            }
            Trace.WriteLine("KDTree.InitializeTree(Datum[]): Finished Creating Tree. Time taken- " + PrintTime(startTime, endTime));
            Trace.WriteLine("No. of Nodes: " + nNodes);
            Trace.Close();
        }


        private unsafe void CreateTreeRecursively(double* dataPtr, int *idPtr, int dataSize, int level)
        {
            
            // check if this node should be a leaf
            if (dataSize <= idealLeafSize)
            {
                // is leaf ... create this as a leaf node and then return
                SetBoundedRegion(dataPtr, dataSize);
                nodeList.Add(new KNode(-1, -1, nNodes, dataSize, level, this));
                nNodes++;
                minPtr += k;
                maxPtr += k;
                return;
            }

            // otherwise split this data into 2. first find where to split
            double splitPoint;
            int splitDimension = GetMaxRangeDimension(dataPtr, idPtr, dataSize, out splitPoint);
            minPtr += k;
            maxPtr += k;
            int leftSize;
            // now split the data
            bool splitOccured = GetLeftRightSplit(
                dataPtr, idPtr, dataSize, splitDimension, splitPoint, out leftSize );

            int rightSplitSize = dataSize - leftSize;

            if (splitOccured && leftSize != 1 && rightSplitSize != 1)
            {
                // then this is not a leaf node
                int nodeIndex = nNodes;
                KNode node = new KNode(splitDimension, splitPoint, nNodes, dataSize, level, this);
                nodeList.Add(node);
                nNodes++;
                level++;
                node.SetLC(nNodes);
                CreateTreeRecursively(dataPtr, idPtr, leftSize, level);
                node.SetRC(nNodes);
                CreateTreeRecursively((dataPtr+(leftSize*k)), idPtr, rightSplitSize, level);
            }
            else
            {
                nodeList.Add(new KNode(-1, -1, nNodes, dataSize, level, this));
                nNodes++;
            }
        }

        private int GetInitialArraySize(int dataSize)
        {
            int baseTwo = (int)Math.Ceiling(Math.Log(
                                
                                    ((double)(2 * dataSize) / idealLeafSize), 2));
            return (int)(2 * Math.Pow(2, baseTwo));
        }

        private unsafe int GetMaxRangeDimension(double *dataPtr, int* idPtr, int dataSize, out double splitPoint)
        {
            SetBoundedRegion(dataPtr, dataSize);
            double maxRange = (*maxPtr - *minPtr);
            int maxRangeIndex = 0;
            for (int i = 1; i < k; ++i)
            {
                double range = *(maxPtr+i) - *(minPtr+i);
                if (range > maxRange)
                {
                    maxRange = range;
                    maxRangeIndex = i;
                }
            }
            splitPoint = ( *(maxPtr+maxRangeIndex) + *(minPtr+maxRangeIndex) ) / 2;
            //Console.WriteLine("SplitPoint:" + splitPoint);
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


        // this bool return true if the plsit makes sense false otherwise
        private unsafe bool GetLeftRightSplit(double *dataPtr, int* idPtr, int dataSize, 
            int splitDimension, double splitPoint, out int leftSize )
        {

            int start = 0;
            int end = dataSize - 1;
            double* rt = dataPtr + ((dataSize-1)*k);
            double* lt = dataPtr;
            int lftIdx = start;
            int rtIdx = end;

            while (true)
            {
                // run upto the point where lftIdx datum should be in the right side
                while (lftIdx <= end && *(lt + splitDimension) <= splitPoint)
                {
                    lt+= k;
                    lftIdx++;
                }

                while (rtIdx >= start && *(rt + splitDimension) > splitPoint)
                {
                    rt-= k;
                    rtIdx--;
                }

                if (lftIdx < rtIdx)
                {
                    double temp;
                    for (int i = 0; i < k; ++i)
                    {
                        temp = *(rt + i);
                        *(rt + i) = *(lt + i);
                        *(lt + i) = temp;
                    }

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
            return idealLeafSize;
        }

        public void InitialLizeTree(String dbName, String tableName)
        {
        }

        // returns true if the tree is Uninitialized
        public bool IsEmpty()
        {
            return nodeList[0] == null ? true : false;
        }

        // returns the max number of children a node can have ie fanout
        public int GetFanOut()
        {
            return fanOut;
        }

        public void SaveToDb(String dbName, String tableName)
        {
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

    }



}
/**
This was the attempt at a non recursive version...
 lets start with recursive and see where we get
 * 
 * 
   int k = data[0].GetDimensionality();
        // create initial array
        nodeList = new ArrayList(GetInitialArraySize(data.Length));

        // now we populate the array and root
        // initially we create the root and its children outside the while
        // this is because in the loop we would be pushing and popping the parent
        // into the parent stack. This is because at the point the child of a parent
        // is popped we need to assign the parent with the value child's index in the array
        // but we cannot do this for the root. Also, this extra part can be avoided
        // by checking everytime if the parent stack is null. This is inefficient and thus
        // this step has been taken

        Stack childStack = new Stack();
        Stack parentStack = new Stack();
        // TODO.. give these stacks initial sizes
        childStack.Push(data);
        int id = 0;
        int arrayIndex;
        int splitDimension;
        double splitPoint;
        BoundingBox box;
        Datum[][] splitData = new Datum[2][];
        Datum[] poppedData;
        KNode node;
        KNode parent;

        //
        poppedData = (Datum[])childStack.Pop();
        splitDimension = GetMaxRangeDimension(poppedData, out splitPoint, out box);
        node = new KNode(id++, splitDimension, splitPoint, box, this);
        // now check for leaf condition
        if (poppedData.Length <= idealLeafSize)
        {
            // is leaf ignore
        }
        else   // try and split the data
        {
            bool splitOccured = GetLeftRightSplit(poppedData, splitDimension, splitPoint, splitData);
            if (splitOccured)
            {
                // not leaf
                childStack.Push(splitData[0]);
                childStack.Push(splitData[1]);
                parentStack.Push(node);
                parentStack.Push(node);
            }
        }
        nodeList.Add(node);
        arrayIndex++;

        if (node.IsLeaf())
        {
            return;
        }
        
        //
        while(childStack.Count != 0)
        {
            poppedData = (Datum[])childStack.Pop();
            splitDimension = GetMaxRangeDimension(poppedData, out splitPoint, out box);
            node = new KNode(id++, splitDimension, splitPoint, box, this);
            // now check for leaf condition
            if (poppedData.Length <= idealLeafSize)
            {
                // is leaf
            }
            else   // try and split the data
            {
                bool splitOccured = GetLeftRightSplit(poppedData, splitDimension, splitPoint, splitData);
                if(splitOccured)
                {
                    // not leaf
                    childStack.Push(splitData[0]);
                    childStack.Push(splitData[1]);

                }
                else
                {
                    // leaf
                }
            }
            parent = parentStack.Pop();
            if (parent.GetLC() == 0)
            {
                parent.SetLC(arrayIndex);
            }
            else
            {
                parent.SetRC(arrayIndex);
            }

            
            nodeList.Add(node);
            arrayIndex++;
            
            // check for leaf condition to be true
        }


*/