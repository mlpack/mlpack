using System;
using System.Collections.Generic;
using System.Text;
using System.IO;
namespace Utilities
{

    public class FileUtilities
    {
        private static int initialDataSize = 5000000;
        private static int initialIdSize = 1000000;

        /// <summary>
        /// Reads a file and returns the data in it. 
        /// Assumes that the first column is the id.
        /// </summary>
        public static void GetFileData(String fileName,
                    out int[] ids, out double[] data, 
                    char[] colSeperater, bool ignoreFirstLine )
        {
            List<double> dataList = new List<double>(initialDataSize);
            List<int> idList = new List<int>(initialIdSize);

            if (!File.Exists(fileName)) 
            {
                throw new FileNotFoundException();
            }
            
            using (StreamReader sr = File.OpenText(fileName))
            {
                String input;
                if (ignoreFirstLine)
                {
                    input = sr.ReadLine();
                    if (input == null)
                    {
                        throw new Exception("File is empty");
                    }
                }
                

                while ((input=sr.ReadLine())!=null) 
                {
                    String[] colVals = input.Split(colSeperater);
                    
                    idList.Add(int.Parse(colVals[0]));
                    for( int i = 1 ; i < colVals.Length; i++ )
                    {
                        dataList.Add(double.Parse(colVals[i]));
                    }
                }
                ids = idList.ToArray();
                data = dataList.ToArray();
                sr.Close();
            }
        }
    }
}
