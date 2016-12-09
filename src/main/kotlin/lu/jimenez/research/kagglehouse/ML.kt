package lu.jimenez.research.kagglehouse

import weka.classifiers.Classifier
import weka.classifiers.trees.M5P
import weka.core.Instances
import java.io.*


object ML {

    @JvmStatic
    fun main(args: Array<String>) {

        val pathToSaveResult: String

        if(args.size!=1){
            pathToSaveResult = javaClass.classLoader.getResource("").path
        }
        else {
            if (File(args[0]).isDirectory) pathToSaveResult=args[0]
            else pathToSaveResult = javaClass.classLoader.getResource("").path
        }

        val instances = InstanceGen("train.csv", "test.csv").generateInstancesPair()

        experiment2Instance(instances, pathToSaveResult ,"M5P")
    }


    fun experiment2Instance(instances: Pair<Instances, Instances>, pathToSaveResult : String ,nameOfExp: String) {


        val startTime = System.currentTimeMillis()

        val classifier = classifier()

        classifier.buildClassifier(instances.first)

        val endTime = System.currentTimeMillis()
        println("Training took " + (endTime - startTime) + " milliseconds")


        val printStream = PrintStream("$pathToSaveResult/$nameOfExp.csv")
        printStream.println( "Id,SalePrice")


        for ((i,instance) in instances.second.withIndex()) {
            printStream.println("${i+1461},${classifier.classifyInstance(instance)}")
        }
        printStream.close()
    }



    fun classifier(): Classifier {
        return M5P()
    }
}