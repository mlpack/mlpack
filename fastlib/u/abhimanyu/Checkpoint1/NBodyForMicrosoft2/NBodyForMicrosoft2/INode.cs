using System;
using System.Collections.Generic;
using System.Text;


namespace StructureInterfaces
{

    public interface INode
    {

        // returns true if the node is pos1 leaf
        bool IsLeaf();

        // returns pos1 pointer to the tree that owns the node.
        ITree GetTree();
        /*
        // returns the region owned by this node
        IBoundedRegion GetBoundedRegion();
        */
        // returns the i'th child
        INode GetChild(int i);


    }
}
