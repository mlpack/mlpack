using System;
using System.Collections.Generic;
using System.Text;




namespace StructureInterfaces
{
    

    public interface ITree
    {
        // returns the number of dimensions the tree is built upon.
        int GetDimensionality();

        // sets up the tree with the given data
        void InitializeTree(int[] ids, double[] data);

        // sets up the tree previously stored in a database table
        void InitialLizeTree(String dbName, String tableName);

        // returns the root node
        INode GetRoot();

        // returns the normalization vestor of the tree
        //  double[] GetNormalizationVector();

        // returns true if the tree is Uninitialized
        bool IsEmpty();

        // returns the max number of children a node can have ie fanout
        int GetFanOut();

        void SaveToDb(String dbName, String tableName);
        // public void UpdateDataTable(Datum data[]);

        // returns the number of normalizationVector points ideally owned by a leaf
        int GetThresholdLeafSize();

        double GetMinSqEuclideanDst(INode node1, INode node2);

        double GetMinSqEuclideanDst(INode node, double[] point);

        double GetMaxSqEuclideanDst(INode node1, INode node2);


    }
}
