using System;
using System.Collections.Generic;
using System.Text;
//using StructureInterfaces;
using System.Data.SqlClient;
using KDTreeStructures;

namespace Utilities
{
    


    public sealed class TreeUtilities
    {
        public static String KDTreeType = "KD";

        public static String MASTER_TABLE_NAME = "SpatialTrees";

        private static String CREATE_MASTER_TABLE =
            " CREATE TABLE Master.dbo." + MASTER_TABLE_NAME + " ( " +
            " DataDb varchar(100), DataTable varchar(100), TreeDbName varchar(100), " +
            " TreeTableName varchar(100) , MappingTableName varchar(100), " +
            " TreeType varchar(20), Dimensionality int, " +
            " Fanout int, NoNodes int, NoPoints int, TreeSpecifics nvarchar(max) )";

        private static String INSERT_INTO_MASTER =
            " INSERT INTO Master.dbo." + MASTER_TABLE_NAME +
            " Values ( '<DataDb>', '<DataTable>' ,'<DatabaseName>', '<TreeTableName>', '<MappingTableName>' ,'<TreeType>', <k> , <fanOut>, <nNodes>, <nPoints>, '<TreeSpecifics>' ) ";

        public static String CREATE_NODE_DATA_MAPPING =
            " Create Table <MappingTableName> ( NodeId int , DatumId int  ) ";

        public static String GET_NODE_DATA_POINTS =
            " Select a.NodeId, b.* From <DataTableString> b join <MappingTableString> a " +
            " on a.DatumId = b.id order by a.NodeId ";
    
        public static void WriteTreeInfoToMaster(TreeInfo treeInfo)
        {
            SqlConnection connection = new SqlConnection();
            //connection.ConnectionString = "Context Connection=true";
            connection.ConnectionString = "Data Source=KLEENE; Initial Catalog=Master; Integrated Security=True;";
            connection.Open();
            String masterStoreQuery = INSERT_INTO_MASTER;
            masterStoreQuery = masterStoreQuery.Replace("<DataDb>", treeInfo.dataDb);
            masterStoreQuery = masterStoreQuery.Replace("<DataTable>", treeInfo.dataTable);
            masterStoreQuery = masterStoreQuery.Replace("<DatabaseName>", treeInfo.treeDb);
            masterStoreQuery = masterStoreQuery.Replace("<TreeTableName>", treeInfo.treeTableName);
            masterStoreQuery = masterStoreQuery.Replace("<MappingTableName>", treeInfo.mappingTable);
            masterStoreQuery = masterStoreQuery.Replace("<TreeType>", treeInfo.treeType);
            masterStoreQuery = masterStoreQuery.Replace("<k>", "" + treeInfo.dimensionality);
            masterStoreQuery = masterStoreQuery.Replace("<fanOut>", "" + treeInfo.fanOut);
            masterStoreQuery = masterStoreQuery.Replace("<nNodes>", "" + treeInfo.nNodes);
            masterStoreQuery = masterStoreQuery.Replace("<nPoints>", "" + treeInfo.nPoints);
            masterStoreQuery = masterStoreQuery.Replace("<TreeSpecifics>", treeInfo.treeSpecifics);
            SqlCommand comm = new SqlCommand();
            comm.Connection = connection;
            comm.CommandText = masterStoreQuery;
            Console.WriteLine(masterStoreQuery);
            comm.ExecuteNonQuery();
            connection.Close();
        }
        
        public static KDTree GetTreeFromTableName(String treeTable)
        {
            SqlConnection connection = new SqlConnection();
            //connection.ConnectionString = "Context Connection=true";
            connection.ConnectionString = "Data Source=KLEENE; Initial Catalog=Master; Integrated Security=True;";
            connection.Open();
            TreeInfo treeInfo = GetTreeInfo(connection,treeTable);
            KDTree tree;
            if (treeInfo.treeType == KDTreeType)
            {
                tree = new KDTree();
                tree.InitializeTree(treeInfo);
            }
            else
            {
                throw new Exception("What type of tree is this? " + treeInfo.treeType);
            }
            // as more types of trees are added they are added here.
            connection.Close();
            return tree;
        }

        /// <summary>
        /// Gets the tree data table name and node to data mapping table names
        /// </summary>
        /// <param name="connection"></param>
        /// <param name="treeTableName"></param>
        /// <param name="mappingTableStr"></param>
        /// <param name="dataTableStr"></param>
        public static TreeInfo GetTreeInfo(SqlConnection connection, String treeTableName)
        {
            String commText =
                " Select top 1 * from Master.dbo." + TreeUtilities.MASTER_TABLE_NAME +
                " where TreeTableName = '" + treeTableName + "' ";
            Console.WriteLine(commText);
            SqlCommand comm = new SqlCommand(commText, connection);
            SqlDataReader rdr = comm.ExecuteReader();
            
            if (rdr.Read())
            {
                TreeInfo treeInfo = new TreeInfo( rdr.GetString(0),
                                            rdr.GetString(1),
                                            rdr.GetString(2),
                                            rdr.GetString(3),
                                            rdr.GetString(4),
                                            rdr.GetString(5),
                                            rdr.GetInt32(6),
                                            rdr.GetInt32(7),
                                            rdr.GetInt32(8),
                                            rdr.GetInt32(9),
                                            rdr.GetString(10));
                rdr.Close();
                return treeInfo;
            }
            else
            {
                throw new Exception("could not find table " + treeTableName);
            }
            
        }

        public static String GetNodeDataSQLQuery(TreeInfo treeInfo)
        {
            String commText = GET_NODE_DATA_POINTS;
            commText = commText.Replace("<DataTableString>", treeInfo.dataDb + ".dbo." + treeInfo.dataTable);
            commText = commText.Replace("<MappingTableString>", treeInfo.treeDb + ".dbo." + treeInfo.mappingTable);
            return commText;
        }
    }

    public class NodeDataMappingReader : SqlBulkCopyReader
    {
        private int[] nodeIds, dataIds;
        private int size;
        private int count;
        private const int fieldCount = 2;



        public NodeDataMappingReader(int[] nIds, int[] dIds)
        {
            size = nIds.Length;
            nodeIds = nIds;
            dataIds = dIds;
            count = -1;
        }


        override public bool Read()
        {
            count++;
            return count < size ? true : false;
        }


        override public object GetValue(int i)
        {
            switch (i)
            {
                case 0:
                    return nodeIds[count];
                case 1:
                    return dataIds[count];
            }
            return null;
        }
        override public int FieldCount
        {
            get { return fieldCount; }
        }

    }

    public sealed class TreeInfo
    {
        public String dataDb;
        public String dataTable;
        public String treeDb;
        public String treeTableName;
        public String mappingTable;
        public String treeType;
        public int dimensionality;
        public int fanOut;
        public int nNodes;
        public int nPoints;
        public String treeSpecifics;

        public TreeInfo(
            String dataDb,
            String dataTable,
            String treeDb,
            String treeTableName,
            String mappingTable,
            String treeType,
            int dimensionality,
            int fanOut,
            int nNodes,
            int nPoints,
            String treeSpecifics)
        {
            this.dataDb = dataDb;
            this.dataTable = dataTable;
            this.treeDb = treeDb;
            this.treeTableName = treeTableName;
            this.mappingTable = mappingTable;
            this.treeType = treeType;
            this.dimensionality = dimensionality;
            this.fanOut = fanOut;
            this.nNodes = nNodes;
            this.nPoints = nPoints;
            this.treeSpecifics = treeSpecifics;
        }
    }
}
