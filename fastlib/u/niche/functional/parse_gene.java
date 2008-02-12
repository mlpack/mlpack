import java.io.*;
import java.util.*;

class parse_gene {
    public static void main(String args[]) {
	
	try {

	    Hashtable<String,String> geneTable =
		new Hashtable<String,String>(10000);
	    
	    BufferedReader br;
	    br = new BufferedReader(new FileReader("genes/orf_data.txt"));
	    
	    String line;
	    
	    while((line = br.readLine()) != null) {
		String[] entries = line.split("\\t");

		geneTable.put(entries[1], entries[0]);
		
	    }
		
	    br.close();

	    String[] phases = {"m_g1_boundary", "g1", "s", "s_g2", "g2_m"};
	    ArrayList<HashSet<String>> orfSets =
		new ArrayList<HashSet<String>>();

	    for(int phaseI = 0; phaseI < phases.length; phaseI++) {

		HashSet<String> orfSet = new HashSet<String>();
		
		br = new BufferedReader(new FileReader("genes/" +
						       phases[phaseI] +
						       ".txt"));
		
		String geneName;
		
		while((geneName = br.readLine()) != null) {
		    String orfName = geneTable.get(geneName);
		    
		    if(orfName != null) {
			orfSet.add(orfName);
		    }
		    else {
			System.out.println("no match found for " + geneName);
		    }
		}
		
		br.close();

		orfSets.add(orfSet);
		
	    }

	    
	    br = new BufferedReader(new FileReader("genes/combined.txt"));

	    PrintWriter pw;
	    pw = new PrintWriter(new FileWriter("genes/combined_phase.txt"));

	    pw.println(br.readLine()); // just copy first line of file

	    while((line = br.readLine()) != null) {
		int tabLocation = line.indexOf("\t");
		String orfName = line.substring(0, tabLocation);

		int matchedPhase = -1;
		for(int phaseI = 0; phaseI < phases.length; phaseI++) {
		    if(orfSets.get(phaseI).contains(orfName)) {
			if(matchedPhase == -1) {
			    matchedPhase = phaseI;
			}
			else {
			    System.out.println(orfName +
					       " produced double match!!");
			    System.exit(1);
			}
		    }
		}

		if(matchedPhase >= 0) {
		    line = phases[matchedPhase] + "\t" + line;
		}
		else {
		    line = "unknown\t" + line;
		}

		pw.println(line);

	    }

	    br.close();
	    pw.close();

	    
	}
	catch(Exception e) {
	    e.printStackTrace();
	}
	
    }
}
