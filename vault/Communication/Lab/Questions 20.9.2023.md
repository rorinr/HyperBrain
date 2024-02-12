- 3D PLI images vs Cyto images (cellbody staines)
	- All Supervised for the beginning
- triplet loss
	- problem at test time: brute force through patches in second image (maybe not necessary because conventional methods good enough for that)
	- idea of decreasing window   
- Transformer and using attention to predict something or so? 





Wie soll das Model mitteilen manche patches nicht zu nutzen? Modell soll sagen dass sich bestimmte patches als feature eignen  -> uncertanty estimation


Cluster im feature space (welche patches führen zur hohen ähnlichkeit)

modell kann den offset vorhersagen wenn ich bspw die ground truth verschiebe und dementsprechend soll das modell nicht 1 sonder 0.9 predicten