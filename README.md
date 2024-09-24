# digital-twin-of-artificial-pancreas

Data is created with "dataCreation.py" to synthetically produce data. This data is used to fit parameters to the model using "parameterFitting.py". 

Table shows parameter names, initial value, and fitted value:

Param | Init | Fitted 
 ----------------
 kjs |  0.034  |  0.03399999254942131
 kxi |  0.025  |  0.024999992552625264
 kgj |  0.067  |  0.06662107758552974
 kjl |  0.007  |  0.007378921411751184
 kgl |  0.02  |  0.021089443028403505
 kxg |  0.018  |  0.017997994592420977
 kxgi |  0.028  |  0.027999181181596368
 n |  0.01  |  0.010051401179515577
 klambda |  0.6  |  0.5029582435900868
 k2 |  22  |  11.655455943817808
 x |  16  |  22.78992216972028

 Innaccuracies are more significant in the GI part of the model, however, they still produce similar data when run through the model. 

![Screenshot](./Pictures/graphs.png)
