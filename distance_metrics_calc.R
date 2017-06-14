library(spatstat)

get_dists = function(x1,y1,x,y){
  sqrt((x1-x)^2 + (y1-y)^2)
}

args = commandArgs(trailingOnly=TRUE)
img_file = args[1]
text_file = paste(img_file, "_results.csv", sep = "")

df = read.csv(text_file)
print(df)

x = df$x
y = df$y
radius = df$petri_radius[1]
petri_x = df$petri_x[1]
petri_y = df$petri_y[1]

min_dist = numeric()
d1 = numeric()
d2 = numeric()
v = numeric()



for (i in 1:length(x)){
  dists = get_dists(x[i],y[i],x,y)
  dists = dists[-i]
  min_dist[i] = min(dists)
  d1[i] = sum(1/dists)
  d2[i] = sum(1/dists^2)
}

pp = ppp(x,y,window = disc(centre = c(petri_x, petri_y), radius = radius))
v = dirichletAreas(pp)

df = cbind(df, data.frame(min_dist, d1, d2, v))
write.csv(df, text_file)