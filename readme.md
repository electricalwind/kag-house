# Kaggle House

This repository is part of the kaggle competition:
[House Prices: Advanced Regression Techniques](https://www.kaggle.com/c/house-prices-advanced-regression-techniques)

## Context

The code was produced during the "Weka data mining tutorial" that took place at APSEC'2016 at the university of Waikato, Hamilton, 6th December 2016.

## Library Used

* Weka 3.9
* Kotlin 1.0.5-2

## Using the project
To use this project with maven:

- mvn install
- mvn exec:java

the produced csv file can be found in target/classes/

To choose the destination folder of the result, modify the following part of the pom.xml file.

```xml

 <plugin>
                <groupId>org.codehaus.mojo</groupId>
                <artifactId>exec-maven-plugin</artifactId>
                <version>1.4.0</version>
                <configuration>

                    <includePluginDependencies>true</includePluginDependencies>
                    <mainClass>lu.jimenez.research.kagglehouse.ML</mainClass>
                    <arguments>
                       <!-- <argument>YourPath
                        </argument> -->
                    </arguments>
                </configuration>
            </plugin>
                    
```

* Uncomment the argument tag
* Replace 'YourPath' by the path of your destination folder.
