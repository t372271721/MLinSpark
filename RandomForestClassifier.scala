// Databricks notebook source
import org.apache.spark.ml.Pipeline  
import org.apache.spark.ml.classification.{RandomForestClassificationModel, RandomForestClassifier}  
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator  
import org.apache.spark.ml.feature.{IndexToString, StringIndexer, VectorIndexer}  
  
// Load data to DataFrame.  
val trainData = spark.read.format("libsvm").load("/FileStore/tables/jui07e9f1498208108508/poker_train-fef7a.libsvm")  
val testDate = spark.read.format("libsvm").load("/FileStore/tables/jui07e9f1498208108508/poker_test-2412b.libsvm")

// Index labels 
val labelIndexer = new StringIndexer()  
  .setInputCol("label")  
  .setOutputCol("indexedLabel")  
  .fit(trainData)  
//  identify categorical features, features with > 4 distinct values are treated as continuous.  
val featureIndexer = new VectorIndexer()  
  .setInputCol("features")  
  .setOutputCol("indexedFeatures")  
  .setMaxCategories(4)  
  .fit(trainData)  
  
// Split data into 70% training and 30% test.  
//val Array(trainingData, testData) = data.randomSplit(Array(0.3, 0.7))  
  
// build classifier  
val rf = new RandomForestClassifier()  
  .setLabelCol("indexedLabel")  
  .setFeaturesCol("indexedFeatures")  
  .setNumTrees(1)  
  .setMaxDepth(10)
  .setImpurity("gini")
  
// Convert indexed labels back to original labels.  
val labelConverter = new IndexToString()  
  .setInputCol("prediction")  
  .setOutputCol("predictedLabel")  
  .setLabels(labelIndexer.labels)  
  
// build pipeline  
val pipeline = new Pipeline()  
  .setStages(Array(labelIndexer, featureIndexer, rf, labelConverter))  
  
// Training  
val model = pipeline.fit(trainData)  
  
// Make predictions.  
val predictions = model.transform(testDate)  
  
// Select example rows to display.  
display(predictions.select("predictedLabel", "label", "features"))
  
// Select (prediction, true label) and compute test error.  
val evaluator = new MulticlassClassificationEvaluator()  
  .setLabelCol("indexedLabel")  
  .setPredictionCol("prediction")  
  .setMetricName("accuracy")
val accuracy = evaluator.evaluate(predictions)
println("Test Error = " + (1.0 - accuracy))
  
val rfModel = model.stages(2).asInstanceOf[RandomForestClassificationModel]  
println("Learned classification forest model:\n" + rfModel.toDebugString) 
