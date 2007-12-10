using System;
using System.Text;
using Utilities;
using System.Collections.Generic;
using System.Data.SqlClient;
using System.IO;


namespace KDTreeStructures
{
    /// <summary>
    /// This class is responsible for quickly writing all the tree nodes to
    /// the tree table.
    /// </summary>
    class KDTreeDataReader : SqlBulkCopyReader
    {
        /**
         * Static members
         */

        private static String CREATE_TREE_TABLE =
            " Create Table <TreeTableName> ( " +
            " NodeId int, SplitDimension int, SplitPoint float(53), " +
            " Level int, NumPoints int, LC int, RC int, " +
            " MinBoxValues nvarchar(max), MaxBoxValues nvarchar(max) )";



        private List<KNode> nodes;          // the nodes of the tree
        double[] mins, maxs;                // bouding regions
        private int k;                      // dimensionality
        private int size;                   // number of nodes

        private int count;                  
        private const int fieldCount = 9;   // number of columns in tree table
        private KNode current;

        public static String GetCreateTreeTableString(String treeTableName)
        {
            String createTableQuery = CREATE_TREE_TABLE;
            return createTableQuery.Replace("<TreeTableName>", treeTableName);
        }

        public KDTreeDataReader(List<KNode> nodeList,
            double[] boxMins, double[] boxMaxs, int k, int dataSize)
        {
            nodes = nodeList;
            mins = boxMins;
            maxs = boxMaxs;
            this.k = k;
            size = dataSize;
            count = -1;
        }


        override public bool Read()
        {
            count++;
            if (count < size)
            {
                current = nodes[count];
                return true;
            }
            return false;
        }

        private String GetMinBoxString()
        {
            StringBuilder str = new StringBuilder("");
            for (int i = 0; i < k - 1; ++i)
            {
                str.Append(mins[k * count + i]);
                str.Append(",");
            }
            str.Append(mins[k * count + k - 1]);
            return str.ToString();
        }

        private String GetMaxBoxString()
        {
            StringBuilder str = new StringBuilder("");
            for (int i = 0; i < k - 1; ++i)
            {
                str.Append(maxs[k * count + i]);
                str.Append(",");
            }
            str.Append(maxs[k * count + k - 1]);
            return str.ToString();
        }

        override public object GetValue(int i)
        {
            switch (i)
            {
                case 0:
                    return current.GetNodeId();
                case 1:
                    return current.GetSplitDimension();
                case 2:
                    return current.GetSplitPoint();
                case 3:
                    return current.GetLevel();
                case 4:
                    return current.GetNumPoints();
                case 5:
                    return current.GetLC();
                case 6:
                    return current.GetRC();
                case 7:
                    return GetMinBoxString();
                case 8:
                    return GetMaxBoxString();
            }
            return null;
        }
        override public int FieldCount
        {
            get { return fieldCount; }
        }

        /// <summary>
        /// Since this class is used by the KDTree to write its node data to the data base
        /// this function mirrors the write functionality to do the opposite, initialize the
        /// tree from a table previously stored by this class.
        /// </summary>
        /// <param name="dbName"></param>
        /// <param name="tableName"></param>
        /// <param name="boxMins"></param>
        /// <param name="boxMaxs"></param>
        /// <param name="nodeList"></param>
        public static void ReadNodes(String dbName, String tableName,
            double[] boxMins, double[] boxMaxs, List<KNode> nodeList, int k, KDTree tree)
        {
            SqlConnection connection = new SqlConnection();
            //connection.ConnectionString = "Context Connection=true";
            connection.ConnectionString = "Data Source=KLEENE; Initial Catalog=Master; Integrated Security=True;";
            connection.Open();
            String queryText = "Select * from " + dbName + ".dbo." + tableName + " order by NodeId ";
            SqlCommand comm = new SqlCommand(queryText, connection);
            SqlDataReader rdr = comm.ExecuteReader();

            int boxIndex = 0;
            Console.WriteLine(boxMaxs.Length + " " + boxMins.Length);
            unsafe
            {
                fixed (double* minPtr = boxMins, maxPtr = boxMaxs)
                {
                    double* mn = minPtr, mx = maxPtr;
                    while (rdr.Read())
                    {
                        KNode node = new KNode(rdr.GetInt32(1), rdr.GetDouble(2), rdr.GetInt32(0),
                            boxIndex, rdr.GetInt32(4), rdr.GetInt32(3), tree);
                        node.SetLC(rdr.GetInt32(5));
                        node.SetRC(rdr.GetInt32(6));

                        String minStr = rdr.GetString(7);
                        String maxStr = rdr.GetString(8);
                        String[] minDoubles = minStr.Split(',');
                        String[] maxDoubles = maxStr.Split(',');
                        try
                        {
                            for (int i = 0; i < k; ++i)
                            {
                                *mn++ = Double.Parse(minDoubles[i]);
                                

                                *mx++ = Double.Parse(maxDoubles[i]);
                                
                            }
                        }
                        catch (Exception e)
                        {
                            Console.WriteLine(minStr);
                            Console.WriteLine(maxStr);
                            throw e;
                        }


                        boxIndex++;
                        nodeList.Add(node);

                    }
                }
            }
        }

    }

    class KDTreeFileWriter
    {
        public static char nodeValSeperator = '|';
        private KDTree tree;
        int k;
        private double[] boxMins, boxMaxs;
        private int[] ids;
        private double[] data;
        private int[] nodeDataMapping;
        public KDTreeFileWriter(KDTree t, double[] mins, double[] maxs, 
            int[] nodeDataMap, int[] id, double[] dt)
        {
            ids = id;
            data = dt;
            nodeDataMapping = nodeDataMap;
            boxMins = mins;
            boxMaxs = maxs;
            tree = t;
            k = tree.GetDimensionality();
        }

        public void SaveTreeToFile(String treeFileName, String nodeDataFileName, bool append)
        {
            if (append)
            {
                if (!File.Exists(treeFileName) || !File.Exists(nodeDataFileName))
                {
                    Console.WriteLine("Files do not exist. Aborting append.");
                    return;
                }
            }
            else
            {
                if (File.Exists(treeFileName) || File.Exists(nodeDataFileName))
                {
                    Console.WriteLine("Files already exist. Aborting write.");
                    return;
                }
            }
            

            // write the tree nodes to a file
            using (StreamWriter sw =new StreamWriter(treeFileName, append))
            {
                if (!append)
                {
                    // print header
                    sw.WriteLine(tree.GetDimensionality() + " "
                    + tree.GetGlobalNumNodes() + " " + tree.GetGlobalDataSize());
                }
                // print global list
                List<KNode> globalList = tree.GetGlobalNodeList();
                for (int i = 0; i < globalList.Count; i++)
                {
                    sw.WriteLine(GetNodeString(globalList[i], tree.GetGlobalBoxMins(), 
                        tree.GetGlobalBoxMaxs(), i));
                }

                // print subtree list
                //Console.WriteLine("i : " + i);
                for (int i = 0; i < tree.GetNumNodes() - globalList.Count; i++)
                {
                        sw.WriteLine(GetNodeString(tree.GetNode(i), boxMins, boxMaxs, i));
                }
                sw.Close();
            }

            // write the nodes and data to a file
            using (StreamWriter sw = new StreamWriter(nodeDataFileName, append))
            {
                for (int i = 0; i < ids.Length; i++)
                {
                    sw.Write(nodeDataMapping[i] + "" + nodeValSeperator + ids[i]);
                    for (int j = 0; j < k; j++)
                    {
                        sw.Write(nodeValSeperator + "" + data[i * k + j]);
                    }
                    sw.WriteLine();
                }
                sw.Close();
            }

        }

        private String GetNodeString(KNode node, double[] boxMins, double[] boxMaxs, int boxIdx)
        {
           
            return
                node.GetNodeId() + "" + nodeValSeperator +
                node.GetSplitDimension() + "" + nodeValSeperator +
                node.GetSplitPoint() + "" + nodeValSeperator +
                node.GetLevel() + "" + nodeValSeperator +
                node.GetNumPoints() + "" + nodeValSeperator +
                node.GetLC() + "" + nodeValSeperator +
                node.GetRC() + "" + nodeValSeperator +
                GetMinBoxString(boxIdx, boxMins, boxMaxs) + "" + nodeValSeperator +
                GetMaxBoxString(boxIdx, boxMins, boxMaxs);

        }

        private String GetMinBoxString(int count, double[] boxMins, double[] boxMaxs)
        {
            StringBuilder str = new StringBuilder("");
             for (int i = 0; i < k - 1; ++i)
                {
                    str.Append(boxMins[k * count + i]);
                    str.Append(",");
                }
                str.Append(boxMins[k * count + k - 1]);
            
            return str.ToString();
        }

        private String GetMaxBoxString(int count, double[] boxMins, double[] boxMaxs)
        {
            StringBuilder str = new StringBuilder("");
            for (int i = 0; i < k - 1; ++i)
            {
                str.Append(boxMaxs[k * count + i]);
                str.Append(",");
            }
            str.Append(boxMaxs[k * count + k - 1]);
            return str.ToString();
        }
    }

    class KDTreeFileReader
    {
        
        public static void GetTreeFromFile(String fileName, out List<KNode> nodeList,
            out double[] boxMins, out double[] boxMaxs, out int k, out int nNodes, out int nPoints, KDTree tree)
        {
            if (!File.Exists(fileName)) 
            {
                throw new FileNotFoundException();
            }
                        
            using (StreamReader sr = File.OpenText(fileName))
            {
                String input = sr.ReadLine();
                String[] lineOneSplit = input.Split(new char[]{' '});
                k = int.Parse(lineOneSplit[0]);
                nNodes = int.Parse(lineOneSplit[1]);
                nPoints = int.Parse(lineOneSplit[2]);
                nodeList = new List<KNode>(nNodes);
                boxMins = new double[nNodes * k];
                boxMaxs = new double[nNodes * k];

                int count = 0;
                while ((input=sr.ReadLine())!=null)  
                {
                    String[] nodedata = input.Split(new char[]{KDTreeFileWriter.nodeValSeperator});
                    int nodeId = int.Parse(nodedata[0]);
                    int splitDimension = int.Parse(nodedata[1]);
                    double splitPoint = double.Parse(nodedata[2]);
                    int level = int.Parse(nodedata[3]);
                    int numPoints = int.Parse(nodedata[4]);
                    int LC = int.Parse(nodedata[5]);
                    int RC = int.Parse(nodedata[6]);
                    ParseAndAddBoxData(nodedata[7], boxMins, count);
                    ParseAndAddBoxData(nodedata[8], boxMaxs, count);
                    KNode node = new KNode(splitDimension, splitPoint, nodeId, (count/k), numPoints, level, tree);
                    count += k;
                    node.SetLC(LC);
                    node.SetRC(RC);
                    nodeList.Add(node);
                }
                sr.Close();
            }
        }

        private static void ParseAndAddBoxData(String boxString, double[] box, int idx)
        {
            String[] boxSplit = boxString.Split(new char[] { ',' });

            for (int i = 0; i < boxSplit.Length; i++)
            {
                box[idx + i] = double.Parse(boxSplit[i]); 
            }
        }
    }
}
