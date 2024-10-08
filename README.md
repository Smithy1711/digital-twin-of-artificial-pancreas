# digital-twin-of-artificial-pancreas

Data is created with "dataCreation.py" to synthetically produce data. This data is used to fit parameters to the model using "parameterFitting.py". 

Table shows parameter names, initial value, and fitted value:

![Screenshot](./Pictures/table.png)

Innaccuracies are more significant in the GI part of the model, however, they still produce similar data when run through the model. 

![Screenshot](./Pictures/graphs.png)


# future thoughts

can I add a data science function that can, given a set of past GI data of a person over a day, decide whether the person should eat glucose, or take insulin, or do nothing, etc by creating a decision tree?

e.g. most times after a big glucose level dip in the late afternoon, this will cause a spike in the evening. A decision tree may be able to give the correct advice when that nuance occurs. 

I will produce some faux data and work on this so it can work alongside the parameter fitting.