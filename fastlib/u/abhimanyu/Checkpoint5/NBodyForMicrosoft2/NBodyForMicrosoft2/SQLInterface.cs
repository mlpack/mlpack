using System;
using System.Collections.Generic;
using System.Text;
using Microsoft.SqlServer.Server;
using Utilities;
using KDTreeStructures;

namespace NBodyForMicrosoft2
{
    /**
     * Assumptions about the database tables.
     * 1. The data tables all have a column called Id wich is the first column in the table.
     * 2. 
     * 
     * 
     */


    /// <summary>
    /// Class contains all info stores in the master tree table.
    /// </summary>
    
    class SQLInterface
    {
        
        
        
        
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
            tree.SaveToDb(sourceDbName, sourceTableName, destDbName, destTableName);
        }
    }

   
}
