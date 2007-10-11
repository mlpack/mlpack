#!/usr/bin/python

#/tmp/garryb/lcdm.txt, 2, 18, 3358.381664, 3469.685367, 0.3610
#/tmp/garryb/lcdm.txt, 4, 18, 1682.555178, 1807.437799, 0.8583
#/tmp/garryb/lcdm.txt, 8, 18, 812.731948, 940.527689, 1.9400
#/tmp/garryb/lcdm.txt, 16, 18, 395.224485, 531.809224, 2.6155
#/tmp/garryb/lcdm.txt, 32, 18, 210.917186, 358.055520, 4.4185
#/tmp/garryb/lcdm.txt, 46, 18, 151.153713, 328.090205, 6.3462

def calc(clock, name, single, multiple, (cputree, withtree)):
  single = single * clock / 3.2 # the single were run on a slower machine
  print "%s, %.0f (%.2f minute)," % (name, single, single / 60),
  util = {}
  for (k, v) in sorted(list(multiple.items())):
    util[k] = single / v / 2.0 / k
    print ("(%d) %.2f," % (k * 2, util[k])),
    last = util[k]
  besttime = single / 90 / last
  print "%.1f (%.2f min) %f" % (besttime, besttime/60, single / besttime),
  # TODO: Really, I should grab the tree-building time!!!
  fx_rpc_overhead = 0.5 * (cputree-1) # fx-rpc sleeps between machines
  treeutil = single / ((withtree - fx_rpc_overhead) * cputree * 2)
  treespeedup = treeutil * 90
  print ", %.2f %.1f" % (treeutil, treespeedup),
  print

# Have to do some futzing with the numbers since the single-processor version ran on a slower clock speed.
# We'll say the first two ran on 3.0 GHz, and the last two ran on 2.8 GHz.
# However, it wouldn't be out of the ordinary for an actual speedul to occur as tasks get smaller...
calc(3.0, "allnn, timit, 2e6", 4967.922, {16:153.315, 32:85.55, 45:65.347}, (45,137.09))
calc(3.0, "kde, redshift, 5e5", 7859.045, {4:924.6, 8:609.58, 16:242.85, 32:119.91, 46:92.04}, (46,118.24))
calc(2.8, "tpc hi, lcdm, 3e6", 7526.365601, {2:1764.339, 4:885.350625, 8:413.688, 16:214.24966, 32:119.66035, 46:86.84515}, (46,144.65))
calc(2.8, "tpc lo, lcdm, 16.8e7", 14313, {
    2:3358.381664,
    4:1682.555178,
    8:812.731948,
    16:395.224485,
    32:210.917186,
    46:151.153713}, (46,328.97))

def over(name, orig, lo, med, hi):
  print "OVER %s: %.2f, %.2f, %.2f" % (orig, lo / orig, med / orig, hi / orig)

over("NN lcdm1000k", 1.53733, 1.532136, 1.580965, 1.947534)
over("NN lcdm100k", 0.155726, 0.15885, 0.162021, 0.203000)
over("NN timit100k", 22.7117, 22.8054, 25.074252, 0.203000)
over("KDE a50k", 102.5, 102.6, 102.9, 102.5)
over("TPClo lcdm100k", 1.986, 2.012, 2.049, 2.125)
over("TPChi lcdm100k", 16.76, 16.89, 17.72, 19.2)

def dualcore(str, orig, dual):
  print "DUAL %s: %.2f" % (str, orig / dual / 2)

dualcore("lcdm1000k.txt", 1.530366, 0.764612)
dualcore("lcdm100k.txt", 0.142097, 0.072364)
dualcore("timit100k.txt", 22.690698, 11.649426)
dualcore("a50k.txt", 102.959103, 49.446226)
dualcore("lcdm100k.txt", 1.985214, 1.021839)
dualcore("lcdm100k.txt", 16.726300, 8.371601)

"""


/local/garryb/lcdm.txt, , 18, 14313.631419, 14421.011832,

/local/garryb/lcdm3000k.txt, , 90, 7526.365601, 7547.323740,
/tmp/garryb/lcdm3000k.txt, 2, 90, 1764.333915, 1784.780000, 1.0000
/tmp/garryb/lcdm3000k.txt, 4, 90, 885.350625, 907.580057, 2.9945
/tmp/garryb/lcdm3000k.txt, 8, 90, 413.688281, 439.047169, 6.9080
/tmp/garryb/lcdm3000k.txt, 16, 90, 214.246966, 244.685013, 13.9476
/tmp/garryb/lcdm3000k.txt, 32, 90, 119.660350, 170.881812, 25.4586
/tmp/garryb/lcdm3000k.txt, 46, 90, 86.845145, 144.649013, 35.0314

/local/garryb/a500k.txt, , 0.05, 7859.045117, 7863.014313,
/tmp/garryb/a500k.txt, 4, 0.05, 924.604805, 930.016924, 2.9995
/tmp/garryb/a500k.txt, 8, 0.05, 609.580028, 635.747685, 6.9503
/tmp/garryb/a500k.txt, 16, 0.05, 242.851925, 254.274229, 14.6071
/tmp/garryb/a500k.txt, 32, 0.05, 119.911340, 139.760265, 28.7479
/tmp/garryb/a500k.txt, 46, 0.05, 82.038028, 118.238711, 38.3729

/tmp/garryb/timit2000k.txt, 16, , 153.314920, 208.486512, 14.6171
/tmp/garryb/timit2000k.txt, 32, , 85.554880, 149.544739, 29.1878
/tmp/garryb/timit2000k.txt, 45, , 65.347369, 137.087579, 40.2455
/local/garryb/timit2000k.txt, , , 4967.922000, 5009.877560,
"""

#/tmp/garryb/timit4000k.txt, 16, , 438.141895, 550.783006, 14.5667
#/tmp/garryb/timit4000k.txt, 32, , 241.590771, 355.503673, 29.0087
#/tmp/garryb/timit4000k.txt, 45, , 182.481596, 296.100844, 40.2777
