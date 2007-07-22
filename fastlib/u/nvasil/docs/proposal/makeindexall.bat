
if EXIST %1.glo makeindex -t %1.glg -o %1.gls -s gtthesis-gloss.ist %1.glo
if EXIST %1.los makeindex -t %1.llg -o %1.lss -s gtthesis-los.ist %1.los
if EXIST %1.idx makeindex -o %1.ind %1.idx
