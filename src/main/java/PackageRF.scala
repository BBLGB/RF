import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.classification.RandomForestClassifier
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.ml.feature.{IndexToString, StringIndexer, VectorAssembler, VectorIndexer}
import org.apache.spark.ml.linalg.{Vector, Vectors}
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.types.DoubleType
import org.apache.spark.sql.{DataFrame, SparkSession}

import java.io.{File, PrintWriter}
import scala.collection.mutable.ArrayBuffer
import scala.util.Random

object PackageRF {
  def main(args: Array[String]): Unit = {
    /**
     * 1.创建SparkSession
     */
    val session = SparkSession.builder().master("spark://192.168.17.10:7077").appName("com.dr.ml").getOrCreate()
    import org.apache.spark.sql.functions._
    import session.implicits._

    /**
     * 2.数据读取
     *
     */
//    val writer = new PrintWriter(new File("hdfs://192.168.17.10:9000/accuracy.txt"))
//    writer.flush();   //清空文件内容
//    writer.println("accuracy = ")
//    writer.close()

    var csvData = session.read
      .option("header", value = true)
      .csv(path = "hdfs://192.168.17.10:9000/package8.csv")
    csvData.show()

    println(csvData.count())

    val ip2Long = (ip: String) => {
      //将IP地址转为Long，这里有固定的算法
      val ips: Array[String] = ip.split("\\.")
      var ipNum: Long = 0L
      for (i <- ips) {
        ipNum = i.toLong | ipNum << 8L
      }
      ipNum
    }
    val ip2LongUDF = udf(ip2Long)

    val threshold = 1000.0

    val appcount = csvData.groupBy("ProtocolName")
      .count()
      .where("count<" + threshold / 100)
      .select("ProtocolName")
      .map(row => row.mkString)
      .collect()
      .toList

    val appdel = appcount.mkString("('", "', '", "')")
    //println(appdel)
    val updatedDf = csvData.withColumn("SourceIP", ip2LongUDF($"Source_IP"))
      .withColumn("DestinationIP", ip2LongUDF($"Destination_IP"))
      .drop("Source_IP", "Destination_IP", "Flow_ID", "Timestamp", "Label")
      .where("ProtocolName not in" + appdel)

    val features = updatedDf.columns
      .filter(_ != "ProtocolName")

    updatedDf.show()

    val cols = updatedDf.columns.map(f => if (features.contains(f)) col(f).cast(DoubleType)
    else col(f))
    val processed_data = updatedDf.select(cols: _*)


    processed_data.show()

    val labelAndVecDF = new VectorAssembler().setInputCols(features).setOutputCol("features")
      .transform(processed_data).select("features", "ProtocolName")

    labelAndVecDF.show()

    val counter = processed_data.groupBy("ProtocolName")
      .count()
    counter.show(50)


    val dict_nearMiss = counter.where("count>" + threshold.toString())
      .select("ProtocolName")
      .map(row => row.mkString)
      .collect()
      .toList
    val dict_smote = counter.where("count<=" + threshold.toString())
      .select("ProtocolName")
      .map(row => row.mkString)
      .collect()
      .toList

    println(dict_nearMiss)
    println(dict_smote)

    var nearMissData = labelAndVecDF.limit(0)
    dict_nearMiss.foreach(protocol => {
      val nearPro = labelAndVecDF.where("ProtocolName = '" + protocol + "'")
      nearMissData = nearMissData.union(nearPro.sample(false, threshold / nearPro.count()))
    })
    nearMissData.show()
    nearMissData.groupBy("ProtocolName").count().show()

    val kNei = 5

    //var smoteData = processed_data.where("ProtocolName in "+dict_smote.mkString("('", "', '", "')"))
    var smoteData = labelAndVecDF.where("ProtocolName in " + dict_smote.mkString("('", "', '", "')"))
    dict_smote.foreach(protocol => {
      val smotePro = labelAndVecDF.where("ProtocolName = '" + protocol + "'")
        .rdd.map(_.getAs[Vector](0))
        .repartition(10)
      val nNei = ((threshold - smotePro.count()) / smotePro.count()).toInt
      val vecRDD: RDD[Vector] = smote(smotePro, kNei, nNei)
      //生成dataframe
      val vecDF: DataFrame = vecRDD.map(vec => (vec, protocol)).toDF("features", "ProtocolName")
      //      val newCols = (0 until features.size).map(i => $"features".getItem(i).alias(features(i)))
      //      val newDF = vecDF.select(newCols: _*).withColumn("ProtocolName", lit(protocol))
      //      newDF.show()
      //      smoteData.show()
      smoteData = smoteData.union(vecDF)
    })

    smoteData.show()
    smoteData.groupBy("ProtocolName").count().show()

    val data = nearMissData.union(smoteData)

    data.show()
    data.groupBy("ProtocolName").count().show()

    //标识整个数据集的标识列和索引列
    val labelIndexer = new StringIndexer()
      .setInputCol("ProtocolName")
      .setOutputCol("indexedLabel")
      .fit(data)
    //设置树的最大层次
    val featureIndexer = new VectorIndexer()
      .setInputCol("features")
      .setOutputCol("indexedFeatures")
      .setMaxCategories(4)
      .fit(data)
    //拆分数据为训练集和测试集（7:3）
    val Array(trainingData, testData) = data.randomSplit(Array(0.7, 0.3))
    testData.show(20)
    //创建模型
    val randomForest = new RandomForestClassifier()
      .setLabelCol("indexedLabel")
      .setFeaturesCol("indexedFeatures")
      .setNumTrees(5)
      .setMaxDepth(5)
    //转化初始数据
    val labelConverter = new IndexToString()
      .setInputCol("prediction")
      .setOutputCol("predictedLabel")
      .setLabels(labelIndexer.labels)
    //使用管道运行转换器和随机森林算法
    val pipeline = new Pipeline()
      .setStages(Array(labelIndexer, featureIndexer, randomForest, labelConverter))
    //训练模型
    val model = pipeline.fit(trainingData)
    //预测
    val predictions = model.transform(testData)
    //输出预测结果
    predictions.select("predictedLabel", "ProtocolName", "features").show(100)
    //创建评估函数，计算错误率
    val evaluator = new MulticlassClassificationEvaluator()
      .setLabelCol("indexedLabel")
      .setPredictionCol("prediction")
      .setMetricName("accuracy")

    val accuracy = evaluator.evaluate(predictions)
    println("accuracy = " + accuracy)



    session.stop()

  }

  def axpy(a: Double, x: Vector, y: Vector): Vector = {
    val ny = new ArrayBuffer[Double]()
    for (i <- 0 until x.size) {
      //println(i,x(i),y(i),a*x(i)+y(i))
      ny.insert(i, a * x(i) + y(i))
    }
    Vectors.dense(ny.toArray)
  }

  //smote
  def smote(data: RDD[Vector], k: Int, N: Int): RDD[Vector] = {
    val vecAndNeis: RDD[(Vector, Array[Vector])] = data.mapPartitions(iter => {
      val vecArr: Array[Vector] = iter.toArray
      //对每个分区的每个vector产生笛卡尔积
      val cartesianArr: Array[(Vector, Vector)] = vecArr.flatMap(vec1 => {
        vecArr.map(vec2 => (vec1, vec2))
      }).filter(tuple => tuple._1 != tuple._2)
      cartesianArr.groupBy(_._1).map { case (vec, vecArr) => {
        (vec, vecArr.sortBy(x => Vectors.sqdist(x._1, x._2)).take(k).map(_._2))
      }
      }.iterator
    })
    //从这k个近邻中随机挑选一个样本，以该随机样本为基准生成N个新样本
    val vecRDD = vecAndNeis.flatMap { case (vec, neighbours) =>
      (1 to N).map { i =>
        val newK = if (k > neighbours.size) neighbours.size else k
        val rn = neighbours(Random.nextInt(newK))
        var diff = rn.copy
        diff = axpy(-1.0, vec, diff)
        var newVec = vec.copy
        newVec = axpy(Random.nextDouble(), diff, newVec)
        newVec
      }.iterator
    }
    vecRDD
  }
}
