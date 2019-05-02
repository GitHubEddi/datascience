from pyspark.sql import SparkSession
from pyspark.ml.regression import LinearRegression
from pyspark.ml.feature import VectorAssembler

import matplotlib.pyplot as plt
import numpy as np

# Create a SparkSession (Note, the config section is only for Windows!)
spark = SparkSession.builder.config("spark.sql.warehouse.dir", "file:///C:/temp").appName("lr").getOrCreate()

# Load up our data
data = spark.read.csv("Salary_Data.csv", inferSchema=True, header=True )
#print('Data')
#data.printSchema()

#Convert independent variables to features
assembler = VectorAssembler(inputCols=['YearsExperience'], outputCol='features')
output = assembler.transform(data)
#print('Output')
#output.printSchema()

final_data = output.select('features','Salary')
#print('final_data')
#final_data.show()

#Split Training and Test data - 70/30
train_data, test_data = final_data.randomSplit([0.7,0.3])

lr = LinearRegression(labelCol='Salary')
lr_model = lr.fit(train_data)

test_results = lr_model.evaluate(test_data)

unlabled = test_data.select('features')
predictions = lr_model.transform(unlabled)
predictions.show()

#accuracy
print('test_results.r2',test_results.r2)

# Convert Spark dataframe to num array
x = np.array(data.select('YearsExperience').collect())
y = np.array(data.select('Salary').collect())
pred = output.select('features')
predictions = lr_model.transform(pred)
pred = np.array(predictions.select('prediction').collect())


#Visualising the Training set results
plt.scatter(x, y, color='red')
plt.plot(x, pred, color='blue')
plt.title('Salary vs Experience ')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()