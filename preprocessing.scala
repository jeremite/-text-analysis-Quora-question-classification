// Databricks notebook source
import org.apache.spark.ml.feature.Tokenizer
import org.apache.spark.ml.feature.RegexTokenizer
import org.apache.spark.ml.feature.StopWordsRemover
import org.apache.spark.ml.feature.NGram
import org.apache.spark.ml.feature.HashingTF
import org.apache.spark.ml.feature.IDF

// COMMAND ----------

// File location and type
val file_location = "/FileStore/tables/train.csv"
val file_type = "csv"

// CSV options
val infer_schema = "false"
val first_row_is_header = "true"
val delimiter = ","

// The applied options are for CSV files. For other file types, these will be ignored.
val df = spark.read.format(file_type)
.option("inferSchema", infer_schema)
.option("header", first_row_is_header)
.option("sep", delimiter)
.load(file_location)

// COMMAND ----------

val df_used =df.filter($"question_text".isNotNull)


val tokenizer = new Tokenizer()
  .setInputCol("text")
  .setOutputCol("words")
val remover = new StopWordsRemover()
  .setInputCol("words")
  .setOutputCol("filteredWords")

val hashingTF = new HashingTF()
  .setNumFeatures(100)
  .setInputCol("filteredWords")
  .setOutputCol("rawFeatures")
val idf = new IDF()
  .setInputCol("rawFeatures")
  .setOutputCol("features")

val pipeline = new Pipeline()
  .setStages(Array(tokenizer,remover, hashingTF, idf))

// Fit the pipeline to training documents.
val model = pipeline.fit(df_used)
val train_out = model.transform(df_used)
  .selectExpr("id","text","features")


/*
// COMMAND ----------

val tokenizer = new Tokenizer().setInputCol("question_text").setOutputCol("words")
val wordsDF = tokenizer.transform(df_used)

// COMMAND ----------

val remover = new StopWordsRemover().setInputCol("words").setOutputCol("filteredWords")
val noStopWordsDF = remover.transform(wordsDF)

// COMMAND ----------

val ngram = new NGram().setN(2).setInputCol("filteredWords").setOutputCol("ngrams")
val nGramDF = ngram.transform(noStopWordsDF)

// COMMAND ----------

val hashingTF = new HashingTF().setInputCol("filteredWords").setOutputCol("rawFeatures").setNumFeatures(100)
val rawFeaturesDF = hashingTF.transform(noStopWordsDF)
val idf = new IDF().setInputCol("rawFeatures").setOutputCol("features")
val idfModel = idf.fit(rawFeaturesDF)
val featuresDF = idfModel.transform(rawFeaturesDF)

// COMMAND ----------

featuresDF.select("question_text","features").show()

// COMMAND ----------

*/
