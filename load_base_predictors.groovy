/*
    datasink: A Pipeline for Large-Scale Heterogeneous Ensemble Learning
    Copyright (C) 2013 Sean Whalen
    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.
    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.
    You should have received a copy of the GNU General Public License
    along with this program.  If not, see [http://www.gnu.org/licenses/].
*/

import java.io.*
import java.text.*
import java.util.*
import java.util.zip.*

import weka.classifiers.*
import weka.classifiers.meta.*
import weka.core.*
import weka.core.converters.ConverterUtils.DataSource
import weka.filters.*
import weka.filters.supervised.instance.*
import weka.filters.unsupervised.attribute.*
import weka.filters.unsupervised.instance.*
import weka.attributeSelection.*
import weka.attributeSelection.ClassifierAttributeEval


void dump(instances, filename) {
    w = new BufferedWriter(new FileWriter(filename))
    w.write(instances.toString())
    w.write("\n")
    w.flush()
    w.close()
}

Instances balance(instances) {
    balanceFilter = new SpreadSubsample()
    balanceFilter.setDistributionSpread(1.0)
    balanceFilter.setInputFormat(instances)
    return Filter.useFilter(instances, balanceFilter)
}

// parse options
parentDir                   = args[0] //path to parent folder of features
rootDir                     = args[1] //path to feature folder
currentFold                 = args[2]
currentBag                  = Integer.valueOf(args[3])
inputFilename		    = rootDir + "/data.arff"
// }
// String[] classifierString   = args[4]
shortClassifierName = args[4]

// load data parameters from properties file
p = new Properties()
p.load(new FileInputStream(parentDir + "/weka.properties"))
workingDir          = rootDir + "/" + p.getProperty("workingDir", ".").trim()
idAttribute         = p.getProperty("idAttribute", "").trim()
classAttribute      = p.getProperty("classAttribute").trim()
// balanceTraining     = Boolean.valueOf(p.getProperty("balanceTraining", "true"))
// balanceTest         = Boolean.valueOf(p.getProperty("balanceTest", "false"))
// assert p.containsKey("foldCount") || p.containsKey("foldAttribute")
// if (p.containsKey("foldCount")) {
//     foldCount       = Integer.valueOf(p.getProperty("foldCount"))
// }
foldAttribute       = p.getProperty("foldAttribute", "").trim()
// nestedFoldCount     = Integer.valueOf(p.getProperty("nestedFoldCount"))
bagCount            = Integer.valueOf(p.getProperty("bagCount"))
// writeModel          = Boolean.valueOf(p.getProperty("writeModel", "false"))

// load data, determine if regression or classification
source              = new DataSource(inputFilename)
data                = source.getDataSet()
regression          = data.attribute(classAttribute).isNumeric()
if (!regression) {
predictClassValue = p.getProperty("predictClassValue").trim()
}


// shuffle data, set class variable
data.randomize(new Random(1))
data.setClass(data.attribute(classAttribute))
cls_gzip_input = new GZIPInputStream(new FileInputStream(new File(classifierDir+"local_model.gz")))
classifier = (Classifier) weka.core.SerializationHelper.read(cls_gzip_input);

outputPrefix = sprintf "validation-%s-%02d", currentFold, currentBag
writer = new PrintWriter(new GZIPOutputStream(new FileOutputStream(new File(classifierDir, outputPrefix + ".csv.gz"))))
header = sprintf "# %s@%s %.2f minutes %s\n", System.getProperty("user.name"), java.net.InetAddress.getLocalHost().getHostName(), durationMinutes, classifierString.join(" ")
writer.write(header)
writer.write("id,label,prediction,fold,bag,classifier\n")

if (foldAttribute != ""){
    for (instance in test) {
        id = instance.stringValue(test.attribute(idAttribute))
        double prediction
        if (!regression) {
    //         label = (instance.stringValue(instance.classAttribute()).equals(predictClassValue)) ? 1 : 0
            prediction = filteredClassifier.distributionForInstance(instance)[predictClassIndex]
        } else {
    //         label = instance.classValue()
            prediction = filteredClassifier.distributionForInstance(instance)[0]
        }
        row = sprintf "%s,%s,%f,%s,%s,%s\n", id, label, prediction, currentFold, currentBag, shortClassifierName
        writer.write(row)
    }
}
writer.flush()
writer.close()



