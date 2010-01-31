using System;
using System.Diagnostics;
using System.Collections;
using System.Collections.Generic;
using StructureInterfaces;


namespace KDTreeStructures
{
    // considering splitting this clas into 2
    public sealed class KNode : INode
    {
        // these are set in constructor
        public double splitPoint;  // -1 if leaf
        public int splitDimension; // -1 if leaf
        private int nodeId;
        // these are set using setters
        private int lcIndex, rcIndex;   // -1 if leaf

        // mr data
        public int numPoints;
        public int level;
        private int boxIdx;
        private KDTree tree;
        
        public KNode(int splitDim, double splitPt, int id,
            int boxIndex, int numPts, int lvl, KDTree ownerTree)
        {
            
            splitDimension = splitDim;
            splitPoint = splitPt;
            nodeId = id;
            boxIdx = boxIndex;//new BoundingBox(boundingBox);
            tree = ownerTree;
            numPoints = numPts;
            level = lvl;
            lcIndex = -1;
            rcIndex = -1;
        }

        public int GetBoxIdx()
        {
            return boxIdx;
        }

        
        
        public ITree GetTree()
        {
            return tree;
        }

        

        public bool IsLeaf()
        {
            return lcIndex == -1 ? true : false;
        }

       
        public INode GetChild(int i)
        {
            if (i == 0)
            {
                return tree.GetNode(lcIndex);
            }
            else if (i == 1)
            {
                return tree.GetNode(rcIndex);
            }
            else
            {
                throw new Exception();
            }

        }

        public int GetLC()
        {
            return lcIndex;
        }

        public int GetRC()
        {
            return rcIndex;
        }

        public void SetLC(int index)
        {
            lcIndex = index;
        }

        public void SetRC(int index)
        {
            rcIndex = index;
        }

        public int GetNumPoints()
        {
            return numPoints;
        }

        public int GetLevel()
        {
            return level;
        }

        public int GetNodeId()
        {
            return nodeId;
        }

        public double GetSplitPoint()
        {
            return splitPoint;
        }

        public int GetSplitDimension()
        {
            return splitDimension;
        }

        public double GetBoxMinValue(int dim)
        {
            return tree.GetBoxMins(boxIdx,dim);
        }
        public double GetBoxMaxValue(int dim)
        {
            return tree.GetBoxMaxs(boxIdx,dim);
        }
    }

}