import java.io.*;
import java.util.*;

class parse_gene {
    public static void main(String args[]) {
	
	try {

	    Hashtable<String,String> geneTable =
		new Hashtable<String,String>(10000);
	    
	    BufferedReader br;
	    br = new BufferedReader(new FileReader("orf_data.txt"));
	    
	    String line;
	    
	    while((line = br.readLine()) != null) {
		String[] entries = line.split("\\t");

		geneTable.put(entries[1], entries[0]);
		
	    }
		
	    br.close();

	    System.out.println("processing phase tags");
	    String[] phases = {"m_g1_boundary", "g1", "s", "s_g2", "g2_m"};
	    ArrayList<HashSet<String>> phaseORFSets =
		new ArrayList<HashSet<String>>();

	    for(int phaseI = 0; phaseI < phases.length; phaseI++) {

		HashSet<String> phaseORFSet = new HashSet<String>();
		
		br = new BufferedReader(new FileReader("phases/" +
						       phases[phaseI] +
						       ".txt"));

		String geneName;
		
		while((geneName = br.readLine()) != null) {
		    String orfName = geneTable.get(geneName);
		    
		    if(orfName != null) {
			phaseORFSet.add(orfName);
		    }
		    else {
			System.out.println("no match found for " + geneName);
		    }
		}
		
		br.close();

		phaseORFSets.add(phaseORFSet);
	    }


	    System.out.println("processing cluster tags");
	    String[] clusters =
		{"CLB2", "CLN2", "Histone", "MAT", "MCM", "MET", "SIC1", "Y"};
	    ArrayList<HashSet<String>> clusterORFSets =
		new ArrayList<HashSet<String>>();

	    for(int clusterI = 0; clusterI < clusters.length; clusterI++) {

		HashSet<String> clusterORFSet = new HashSet<String>();
		
		br = new BufferedReader(new FileReader("clusters/" +
						       clusters[clusterI] +
						       ".txt"));

		String geneName;
		
		while((geneName = br.readLine()) != null) {
		    String orfName = geneTable.get(geneName);
		    
		    if(orfName != null) {
			clusterORFSet.add(orfName);
		    }
		    else {
			System.out.println("no match found for " + geneName);
		    }
		}
		
		br.close();

		clusterORFSets.add(clusterORFSet);
	    }


	    
	    br = new BufferedReader(new FileReader("combined.txt"));

	    PrintWriter pw;
	    pw = new PrintWriter(new FileWriter("combined_phase_cluster.txt"));

	    pw.println(br.readLine()); // just copy first line of file

	    while((line = br.readLine()) != null) {
		int tabLocation = line.indexOf("\t");
		String orfName = line.substring(0, tabLocation);

		int matchedPhase = -1;
		for(int phaseI = 0; phaseI < phases.length; phaseI++) {
		    if(phaseORFSets.get(phaseI).contains(orfName)) {
			if(matchedPhase == -1) {
			    matchedPhase = phaseI;
			}
			else {
			    System.out.println(orfName + " produced double match!!");
			    System.exit(1);
			}
		    }
		}

		int matchedCluster = -1;
		for(int clusterI = 0; clusterI < clusters.length; clusterI++) {
		    if(clusterORFSets.get(clusterI).contains(orfName)) {
			if(matchedCluster == -1) {
			    matchedCluster = clusterI;
			}
			else {
			    System.out.println(orfName + " produced double match!!");
			    System.exit(1);
			}
		    }
		}


		String phaseString;
		if(matchedPhase >= 0) {
		    phaseString = phases[matchedPhase];
		}
		else {
		    phaseString = "unknown";
		}

		String clusterString;
		if(matchedCluster >= 0) {
		    clusterString = clusters[matchedCluster];
		}
		else {
		    clusterString = "unknown";
		}
		
		line = phaseString + "\t" + clusterString + "\t" + line;

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
