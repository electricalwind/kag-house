package lu.jimenez.research.kagglehouse

import com.opencsv.CSVReader
import weka.core.*
import java.io.FileReader
import java.util.*


class InstanceGen(trainingFile: String, testingFilePath: String) {
    val features: Array<String>
    private val traincsv: CSVReader
    private val testcsv: CSVReader

    fun generateInstancesPair(): Pair<Instances, Instances> {
        val fv = buildFeatureVector()
        return Pair(generateInstances(fv, true), generateInstances(fv, false))
    }

    private fun buildFeatureVector(): ArrayList<Attribute> {
        val featureVector = arrayListOf<Attribute>()
        for (feature in features) {
            when (feature) {
            //Numeric
            //"Id",
                "MiscVal",
                "MoSold",
                "YrSold",
                "LotFrontage",
                "LotArea",
                "YearBuilt",
                "YearRemodAdd",
                "MasVnrArea",
                "BsmtFinSF1",
                "BsmtFinSF2",
                "BsmtUnfSF",
                "TotalBsmtSF",
                "1stFlrSF",
                "2ndFlrSF",
                "LowQualFinSF",
                "GrLivArea",
                "BsmtFullBath",
                "BsmtHalfBath",
                "FullBath",
                "HalfBath",
                "BedroomAbvGr",
                "KitchenAbvGr",
                "TotRmsAbvGrd",
                "Fireplaces",
                "GarageYrBlt",
                "GarageCars",
                "GarageArea",
                "WoodDeckSF",
                "OpenPorchSF",
                "EnclosedPorch",
                "3SsnPorch",
                "ScreenPorch",
                "PoolArea",
                "SalePrice" -> featureVector.add(Attribute(feature))

            //Ordinal
                "LotShape",
                "Utilities",
                "LandSlope",
                "OverallQual",
                "OverallCond",
                "ExterQual",
                "ExterCond",
                " BsmtQual",
                "BsmtCond",
                "BsmtExposure",
                "BsmtFinType1",
                "BsmtFinType2",
                "HeatingQC",
                "KitchenQual",
                "Functional",
                "FireplaceQu",
                "GarageFinish",
                "GarageQual",
                "GarageCond",
                "PavedDrive",
                "PoolQC",
                "Fence",
                "BsmtQual" -> featureVector.add(Attribute(feature)) //transform ordinal into numerical ;)

            //Nominal
                "MSSubClass" -> {
                    val my_class_values = ArrayList<String>(16)
                    my_class_values.add("20")
                    my_class_values.add("30")
                    my_class_values.add("40")
                    my_class_values.add("45")
                    my_class_values.add("50")
                    my_class_values.add("60")
                    my_class_values.add("70")
                    my_class_values.add("75")
                    my_class_values.add("80")
                    my_class_values.add("85")
                    my_class_values.add("90")
                    my_class_values.add("120")
                    my_class_values.add("150")
                    my_class_values.add("160")
                    my_class_values.add("180")
                    my_class_values.add("190")
                    featureVector.add(Attribute(feature, my_class_values))
                }
                "MSZoning" -> {
                    val my_class_values = ArrayList<String>(9)
                    my_class_values.add("A")
                    my_class_values.add("C (all)")
                    my_class_values.add("FV")
                    my_class_values.add("I")
                    my_class_values.add("RH")
                    my_class_values.add("RL")
                    my_class_values.add("RP")
                    my_class_values.add("RM")
                    my_class_values.add("NA")
                    featureVector.add(Attribute(feature, my_class_values))
                }
                "Street" -> {
                    val my_class_values = ArrayList<String>(2)
                    my_class_values.add("Grvl")
                    my_class_values.add("Pave")
                    featureVector.add(Attribute(feature, my_class_values))
                }
                "Alley" -> {
                    val my_class_values = ArrayList<String>(3)
                    my_class_values.add("Grvl")
                    my_class_values.add("Pave")
                    my_class_values.add("NA")
                    featureVector.add(Attribute(feature, my_class_values))
                }
                "LandContour" -> {
                    val my_class_values = ArrayList<String>(4)
                    my_class_values.add("Lvl")
                    my_class_values.add("Bnk")
                    my_class_values.add("HLS")
                    my_class_values.add("Low")
                    featureVector.add(Attribute(feature, my_class_values))
                }
                "LotConfig" -> {
                    val my_class_values = ArrayList<String>(5)
                    my_class_values.add("Inside")
                    my_class_values.add("Corner")
                    my_class_values.add("CulDSac")
                    my_class_values.add("FR2")
                    my_class_values.add("FR3")
                    featureVector.add(Attribute(feature, my_class_values))
                }
                "Neighborhood" -> {
                    val my_class_values = ArrayList<String>(28)
                    my_class_values.add("Blmngtn")
                    my_class_values.add("Blueste")
                    my_class_values.add("BrDale")
                    my_class_values.add("BrkSide")
                    my_class_values.add("ClearCr")
                    my_class_values.add("CollgCr")
                    my_class_values.add("Crawfor")
                    my_class_values.add("Edwards")
                    my_class_values.add("Gilbert")
                    my_class_values.add("Greens")
                    my_class_values.add("GrnHill")
                    my_class_values.add("IDOTRR")
                    my_class_values.add("Landmrk")
                    my_class_values.add("MeadowV")
                    my_class_values.add("Mitchel")
                    my_class_values.add("NAmes")
                    my_class_values.add("NoRidge")
                    my_class_values.add("NPkVill")
                    my_class_values.add("NridgHt")
                    my_class_values.add("NWAmes")
                    my_class_values.add("OldTown")
                    my_class_values.add("SWISU")
                    my_class_values.add("Sawyer")
                    my_class_values.add("SawyerW")
                    my_class_values.add("Somerst")
                    my_class_values.add("StoneBr")
                    my_class_values.add("Timber")
                    my_class_values.add("Veenker")
                    featureVector.add(Attribute(feature, my_class_values))
                }
                "Condition1", "Condition2" -> {
                    val my_class_values = ArrayList<String>(9)
                    my_class_values.add("Artery")
                    my_class_values.add("Feedr")
                    my_class_values.add("Norm")
                    my_class_values.add("RRNn")
                    my_class_values.add("RRAn")
                    my_class_values.add("PosN")
                    my_class_values.add("PosA")
                    my_class_values.add("RRNe")
                    my_class_values.add("RRAe")
                    featureVector.add(Attribute(feature, my_class_values))
                }
                "BldgType" -> {
                    val my_class_values = ArrayList<String>(5)
                    my_class_values.add("1Fam")
                    my_class_values.add("2fmCon")
                    my_class_values.add("Duplex")
                    my_class_values.add("TwnhsE")
                    my_class_values.add("TwnhsI")
                    featureVector.add(Attribute(feature, my_class_values))
                }
                "HouseStyle" -> {
                    val my_class_values = ArrayList<String>(8)
                    my_class_values.add("1Story")
                    my_class_values.add("1.5Fin")
                    my_class_values.add("1.5Unf")
                    my_class_values.add("2Story")
                    my_class_values.add("2.5Fin")
                    my_class_values.add("2.5Unf")
                    my_class_values.add("SFoyer")
                    my_class_values.add("SLvl")
                    featureVector.add(Attribute(feature, my_class_values))
                }
                "RoofStyle" -> {
                    val my_class_values = ArrayList<String>(6)
                    my_class_values.add("Flat")
                    my_class_values.add("Gable")
                    my_class_values.add("Gambrel")
                    my_class_values.add("Hip")
                    my_class_values.add("Mansard")
                    my_class_values.add("Shed")
                    featureVector.add(Attribute(feature, my_class_values))
                }
                "RoofMatl" -> {
                    val my_class_values = ArrayList<String>(8)
                    my_class_values.add("ClyTile")
                    my_class_values.add("CompShg")
                    my_class_values.add("Membran")
                    my_class_values.add("Metal")
                    my_class_values.add("Roll")
                    my_class_values.add("Tar&Grv")
                    my_class_values.add("WdShake")
                    my_class_values.add("WdShngl")
                    featureVector.add(Attribute(feature, my_class_values))
                }
                "Exterior1st", "Exterior2nd" -> {
                    val my_class_values = ArrayList<String>(18)
                    my_class_values.add("AsbShng")
                    my_class_values.add("AsphShn")
                    my_class_values.add("BrkComm")
                    my_class_values.add("BrkFace")
                    my_class_values.add("CBlock")
                    my_class_values.add("CemntBd")
                    my_class_values.add("HdBoard")
                    my_class_values.add("ImStucc")
                    my_class_values.add("MetalSd")
                    my_class_values.add("Other")
                    my_class_values.add("Plywood")
                    my_class_values.add("PreCast")
                    my_class_values.add("Stone")
                    my_class_values.add("Stucco")
                    my_class_values.add("VinylSd")
                    my_class_values.add("Wd Sdng")
                    my_class_values.add("WdShing")
                    my_class_values.add("NA")
                    featureVector.add(Attribute(feature, my_class_values))
                }
                "MasVnrType" -> {
                    val my_class_values = ArrayList<String>(6)
                    my_class_values.add("BrkCmn")
                    my_class_values.add("BrkFace")
                    my_class_values.add("CBlock")
                    my_class_values.add("None")
                    my_class_values.add("Stone")
                    my_class_values.add("NA")
                    featureVector.add(Attribute(feature, my_class_values))
                }
                "Foundation" -> {
                    val my_class_values = ArrayList<String>(6)
                    my_class_values.add("BrkTil")
                    my_class_values.add("CBlock")
                    my_class_values.add("PConc")
                    my_class_values.add("Slab")
                    my_class_values.add("Stone")
                    my_class_values.add("Wood")
                    featureVector.add(Attribute(feature, my_class_values))
                }
                "Heating" -> {
                    val my_class_values = ArrayList<String>(6)
                    my_class_values.add("Floor")
                    my_class_values.add("GasA")
                    my_class_values.add("GasW")
                    my_class_values.add("Grav")
                    my_class_values.add("OthW")
                    my_class_values.add("Wall")
                    featureVector.add(Attribute(feature, my_class_values))
                }
                "CentralAir" -> {
                    val my_class_values = ArrayList<String>(2)
                    my_class_values.add("N")
                    my_class_values.add("Y")
                    featureVector.add(Attribute(feature, my_class_values))
                }
                "Electrical" -> {
                    val my_class_values = ArrayList<String>(6)
                    my_class_values.add("SBrkr")
                    my_class_values.add("FuseA")
                    my_class_values.add("FuseF")
                    my_class_values.add("FuseP")
                    my_class_values.add("Mix")
                    my_class_values.add("NA")
                    featureVector.add(Attribute(feature, my_class_values))
                }
                "GarageType" -> {
                    val my_class_values = ArrayList<String>(7)
                    my_class_values.add("2Types")
                    my_class_values.add("Attchd")
                    my_class_values.add("Basment")
                    my_class_values.add("BuiltIn")
                    my_class_values.add("CarPort")
                    my_class_values.add("Detchd")
                    my_class_values.add("NA")
                    featureVector.add(Attribute(feature, my_class_values))
                }
                "MiscFeature" -> {
                    val my_class_values = ArrayList<String>(6)
                    my_class_values.add("Elev")
                    my_class_values.add("Gar2")
                    my_class_values.add("Othr")
                    my_class_values.add("Shed")
                    my_class_values.add("TenC")
                    my_class_values.add("NA")
                    featureVector.add(Attribute(feature, my_class_values))
                }
                "SaleType" -> {
                    val my_class_values = ArrayList<String>(11)
                    my_class_values.add("WD")
                    my_class_values.add("CWD")
                    my_class_values.add("VWD")
                    my_class_values.add("New")
                    my_class_values.add("COD")
                    my_class_values.add("Con")
                    my_class_values.add("ConLw")
                    my_class_values.add("ConLI")
                    my_class_values.add("ConLD")
                    my_class_values.add("Oth")
                    my_class_values.add("NA")
                    featureVector.add(Attribute(feature, my_class_values))
                }
                "SaleCondition" -> {
                    val my_class_values = ArrayList<String>(6)
                    my_class_values.add("Normal")
                    my_class_values.add("Abnorml")
                    my_class_values.add("AdjLand")
                    my_class_values.add("Alloca")
                    my_class_values.add("Family")
                    my_class_values.add("Partial")
                    featureVector.add(Attribute(feature, my_class_values))
                }
                else -> {
                    println("$feature was not consider")
                    //featureVector.add(Attribute(feature, listOf()))
                }
            }
        }
        return featureVector
    }

    private fun generateInstances(featureVector: ArrayList<Attribute>, training: Boolean): Instances {
        // declaring variable
        val workingreader: CSVReader
        val name: String

        if (training) {
            workingreader = traincsv
            name = "train"
        } else {
            workingreader = testcsv
            name = "test"
        }
        val instances = Instances(name, featureVector, 0)
        val listofIntance = mutableListOf<Instance>()
        var nextLine = workingreader.readNext()


        while (nextLine != null) {
            val instance = DenseInstance(featureVector.size)

            for ((inc, feature) in featureVector.withIndex()) {
                //handling missing sale price of the testing set

                //ignoring ID field
                val i = inc + 1

                if (i == featureVector.size - 1 && !training) break

                //handming spec/data difference
                when (nextLine[i]) {
                    "Wd Shng" -> nextLine[i] = "Wd Sdng"
                    "CmentBd" -> nextLine[i] = "CemntBd"
                    "Twnhs" -> nextLine[i] = "TwnhsI"
                    "Brk Cmn" -> nextLine[i] = "BrkComm"
                    else -> null
                }
                when (feature.name()) {

                //Nominal
                    "MSSubClass",
                    "Street",
                    "Alley",
                    "MSZoning",
                    "LandContour",
                    "LotConfig",
                    "Neighborhood",
                    "Condition1",
                    "Condition2",
                    "BldgType",
                    "HouseStyle",
                    "RoofStyle",
                    "RoofMatl",
                    "Exterior1st",
                    "Exterior2nd",
                    "MasVnrType",
                    "Foundation",
                    "CentralAir",
                    "Electrical",
                    "SaleType",
                    "SaleCondition",
                    "MiscFeature",
                    "GarageType",
                    "Heating" -> instance.setValue(feature, nextLine[i])

                //Numeric
                //"Id",
                    "LotFrontage",
                    "LotArea",
                    "OverallQual",
                    "OverallCond",
                    "YearBuilt",
                    "YearRemodAdd",
                    "MasVnrArea",
                    "BsmtFinSF1",
                    "BsmtFinSF2",
                    "BsmtUnfSF",
                    "TotalBsmtSF",
                    "1stFlrSF",
                    "2ndFlrSF",
                    "LowQualFinSF",
                    "GrLivArea",
                    "BsmtFullBath",
                    "BsmtHalfBath",
                    "FullBath",
                    "HalfBath",
                    "BedroomAbvGr",
                    "KitchenAbvGr",
                    "TotRmsAbvGrd",
                    "Fireplaces",
                    "GarageCars",
                    "GarageArea",
                    "GarageYrBlt",
                    "WoodDeckSF",
                    "OpenPorchSF",
                    "EnclosedPorch",
                    "3SsnPorch",
                    "ScreenPorch",
                    "PoolArea",
                    "MiscVal",
                    "MoSold",
                    "YrSold",
                    "SalePrice" -> try {
                        val d = nextLine[i].toDouble()
                        instance.setValue(feature, d)
                    } catch (e: NumberFormatException) {
                        instance.setValue(feature, Double.NaN)
                    }

                //Ordinal
                    "LotShape" -> {
                        val ord: Double
                        when (nextLine[i]) {
                            "Reg" -> ord = 0.0
                            "IR1" -> ord = 1.0
                            "IR2" -> ord = 2.0
                            "IR3" -> ord = 3.0
                            else -> ord = 10.0
                        }
                        instance.setValue(feature, ord)
                    }
                    "Utilities" -> {
                        val ord: Double
                        when (nextLine[i]) {
                            "AllPub" -> ord = 0.0
                            "NoSewr" -> ord = 1.0
                            "NoSeWa" -> ord = 2.0
                            "ELO" -> ord = 3.0
                            else -> ord = 10.0
                        }
                        instance.setValue(feature, ord)
                    }
                    "LandSlope" -> {
                        val ord: Double
                        when (nextLine[i]) {
                            "Gtl" -> ord = 0.0
                            "Mod" -> ord = 1.0
                            "Sev" -> ord = 2.0
                            else -> ord = 10.0
                        }
                        instance.setValue(feature, ord)
                    }
                    "ExterQual",
                    "ExterCond",
                    "BsmtQual",
                    "BsmtCond",
                    "HeatingQC",
                    "KitchenQual",
                    "FireplaceQu",
                    "GarageQual",
                    "GarageCond" -> {
                        val ord: Double
                        when (nextLine[i]) {
                            "Ex" -> ord = 0.0
                            "Gd" -> ord = 1.0
                            "TA" -> ord = 2.0
                            "Fa" -> ord = 3.0
                            "Po" -> ord = 4.0
                            "NA" -> ord = 5.0
                            else -> ord = 10.0
                        }
                        instance.setValue(feature, ord)
                    }
                    "BsmtExposure" -> {
                        val ord: Double
                        when (nextLine[i]) {
                            "Gd" -> ord = 0.0
                            "Av" -> ord = 1.0
                            "Mn" -> ord = 2.0
                            "No" -> ord = 3.0
                            "NA" -> ord = 4.0
                            else -> ord = 10.0
                        }
                        instance.setValue(feature, ord)
                    }
                    "BsmtFinType1", "BsmtFinType2" -> {
                        val ord: Double
                        when (nextLine[i]) {
                            "GLQ" -> ord = 0.0
                            "ALQ" -> ord = 1.0
                            "BLQ" -> ord = 2.0
                            "Rec" -> ord = 3.0
                            "LwQ" -> ord = 4.0
                            "Unf" -> ord = 5.0
                            "NA" -> ord = 6.0
                            else -> ord = 10.0
                        }
                        instance.setValue(feature, ord)
                    }
                    "Functional" -> {
                        val ord: Double
                        when (nextLine[i]) {
                            "Typ" -> ord = 0.0
                            "Min1" -> ord = 1.0
                            "Min2" -> ord = 2.0
                            "Mod" -> ord = 3.0
                            "Maj1" -> ord = 4.0
                            "Maj2" -> ord = 5.0
                            "Sev" -> ord = 6.0
                            "Sal" -> ord = 7.0
                            else -> ord = 10.0
                        }
                        instance.setValue(feature, ord)
                    }
                    "GarageFinish" -> {
                        val ord: Double
                        when (nextLine[i]) {
                            "Fin" -> ord = 0.0
                            "RFn" -> ord = 1.0
                            "Unf" -> ord = 2.0
                            "NA" -> ord = 3.0
                            else -> ord = 10.0
                        }
                        instance.setValue(feature, ord)
                    }
                    "PavedDrive" -> {
                        val ord: Double
                        when (nextLine[i]) {
                            "Y" -> ord = 0.0
                            "P" -> ord = 1.0
                            "N" -> ord = 2.0
                            else -> ord = 10.0
                        }
                        instance.setValue(feature, ord)
                    }
                    "PoolQC", "Fence" -> {
                        val ord: Double
                        when (nextLine[i]) {
                            "Ex" -> ord = 0.0
                            "Gd" -> ord = 1.0
                            "TA" -> ord = 2.0
                            "Fa" -> ord = 3.0
                            "NA" -> ord = 4.0
                            else -> ord = 10.0
                        }
                        instance.setValue(feature, ord)
                    }

                    else -> {
                        println(feature.name())
                        //instance.setValue(feature, nextLine[i])
                    }
                }
            }

            listofIntance.add(instance)
            nextLine = workingreader.readNext()
        }

        workingreader.close()

        instances.setClassIndex(featureVector.size - 1)
        instances.addAll(listofIntance)

        return instances
    }

    init {
        val trainingFile = javaClass.classLoader.getResource(trainingFile)
        val testingFile = javaClass.classLoader.getResource(testingFilePath)
        traincsv = CSVReader(FileReader(trainingFile.path))
        testcsv = CSVReader(FileReader(testingFile.path))
        features = traincsv.readNext()
        testcsv.readNext()
    }


}