library("scmamp")
library("ggplot2")
library("Rgraphviz")

args=(commandArgs(TRUE))

if(length(args)==0){
    print("No arguments supplied.")
    ##supply default values
    cd_fn = "../cd_input.csv"
}else{
    for(i in 1:length(args)){
         eval(parse(text=args[[i]]))
    }
}

cd_input <- read.csv(file = cd_fn)

png(file=gsub("csv", "png", cd_fn))
plotCD(cd_input, alpha=0.05, cex=1.25)
dev.off()
