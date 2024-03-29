library("scmamp")
library("ggplot2")
library("Rgraphviz")
library("dplyr")
# library("PMCMR")
library("PMCMRplus")

args=(commandArgs(TRUE))
print(R.version)

if(length(args)==0){
    print("No arguments supplied.")
    ##supply default values
    cd_fn = "./plot_17May/plot/cd_csv/cd_input_go_pmax.csv"
}else{
    for(i in 1:length(args)){
         eval(parse(text=args[[i]]))
    }
}

cd_input <- read.csv(file = cd_fn)
cd_input <- na.omit(cd_input)
fn <- gsub("csv", "pdf", cd_fn)
print(cd_input)
# pv_mat <- posthoc.friedman.nemenyi.test(as.matrix(cd_input))
pv_mat <- frdAllPairsNemenyiTest(as.matrix(cd_input))
p_value_csv_fn <- gsub("cd_input_", "cd_pval_", cd_fn)
write.csv(pv_mat$p.value, p_value_csv_fn)


# fn <- gsub("cd_input", "./plot/cd_input", fn)
prefix = './plot/'
# fn <- strsplit(fn, '/')
fn <- gsub("^.*/", "", fn)
fn <- paste(prefix, fn, sep="")
print(fn)
pdf(file=fn, width=15, bg="white")
# par(mar=c(1,5,1,5))
cdplot <- plotCD(cd_input, alpha=0.05, cex=0.9)  +
  geom_point() +
  coord_fixed(ratio = 0.5) + coord_flip(clip = "off")
# ggsave(filename=gsub("csv", "png", cd_fn), plot=cdplot)
dev.off()


