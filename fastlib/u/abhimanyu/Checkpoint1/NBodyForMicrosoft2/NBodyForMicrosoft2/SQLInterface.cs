using System;
using System.Collections.Generic;
using System.Text;
using Microsoft.SqlServer.Server;
using Utilities;
using KDTreeStructures;

namespace NBodyForMicrosoft2
{
    class SQLInterface
    {
        private static String CREATE_MASTER_TABLE =
            " CREATE TABLE Master.dbo." + MASTER_TABLE_NAME + " ( " +
            " DbName varchar(100), TableName varchar(100) , " +
            " TreeType varchar(20), Dimensionality int, " +
            " Fanout int, NoNodes int, TreeSpecifics nvarchar(max) )";


        public static String MASTER_TABLE_NAME = "SpatialTrees";
        [SqlFunction(Name = "CreateKDTree",
        DataAccess = DataAccessKind.Read,
        IsDeterministic = true,
        IsPrecise = true)]
        public static void CreateKdTree(String sourceDbName, 
            String sourceTableName, String destDbName, String destTableName)
        {
            int[] ids;
            double[] data;
            DatabaseUtilities.GetTableData(sourceDbName, sourceTableName, out ids, out data);
            KDTree tree = new KDTree();
            tree.InitializeTree(ids, data);
            tree.SaveToDb(destDbName, destTableName);
        }
    }
}
