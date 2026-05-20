set.seed(1)
require(MCMCprecision)
require(dplyr)
require(tidyr)
library(ggplot2)
library(cowplot)
library(RColorBrewer)
library(gtable)
library(grid)
library(parallel)


plotfolder="/gammajustification"


############# Functions ############

###################################################
#### Shift Legend Into Empty Facets If Any ########
###################################################
## Base function from: https://stackoverflow.com/questions/54438495/shift-legend-into-empty-facets-of-a-faceted-plot-in-ggplot2
## Provided by Z. Lin and editted by double-beep on Jan 30, 2019
## Function then corrected manually and with the assistance of Claude.ai.
shift_legend <- function(p){
  
  # check if p is a valid object
  if(!inherits(p,"gtable")){
    if(inherits(p,"ggplot")){
      gp <- ggplotGrob(p) # convert to grob
    } else {
      message("This is neither a ggplot object nor a grob generated from ggplotGrob. Returning original plot.")
      return(p)
    }
  } else {
    gp <- p
  }
  
  # check for unfilled facet panels
  facet.panels <- grep("^panel", gp[["layout"]][["name"]])
  empty.facet.panels <- sapply(facet.panels, function(i) "zeroGrob" %in% class(gp[["grobs"]][[i]]))
  empty.facet.panels <- facet.panels[empty.facet.panels]
  if(length(empty.facet.panels) == 0){
    message("There are no unfilled facet panels to shift legend into. Returning original plot.")
    return(p)
  }
  
  # establish extent of unfilled facet panels (including any axis cells in between)
  empty.facet.panels <- gp[["layout"]][empty.facet.panels, ]
  empty.facet.panels <- list(t=min(empty.facet.panels[["t"]]), 
                             l=min(empty.facet.panels[["l"]]),
                             b=max(empty.facet.panels[["b"]]), 
                             r=max(empty.facet.panels[["r"]]))
  
  # extract legend & copy over to location of unfilled facet panels
  guide.idx=grep("guide-box",gp[['layout']][['name']])
  if(length(guide.idx) == 0){
    message("There is no legend (guide-box) present. Returning original plot.")
    return(p)
  }
  # If multiple guide-box entries, pick the one whose grob isn't a zeroGrob
  guide.idx <- guide.idx[!sapply(guide.idx, function(i) inherits(gp$grobs[[i]], "zeroGrob"))]
  if (length(guide.idx) == 0) {
    message("Legend exists in layout but is empty (zeroGrob). Returning original.")
    return(p)
  }
  
  guide.idx  <- guide.idx[1]  # take first non-empty match
  guide.grob <- gp$grobs[[guide.idx]]
  guide.pos  <- gp$layout[guide.idx, ]
  
  gp <- gtable_add_grob(x = gp,
                        grobs = guide.grob,
                        t = empty.facet.panels[["t"]],
                        l = empty.facet.panels[["l"]],
                        b = empty.facet.panels[["b"]],
                        r = empty.facet.panels[["r"]],
                        name = "new-guide-box")
  
  # squash the original guide box's row / column (whichever applicable)
  # & empty its cell
  # Squash the original guide-box row/col to zero size
  if (guide.pos$l == guide.pos$r) {
    gp$widths[guide.pos$l] <- unit(0, "pt")
  }
  if (guide.pos$t == guide.pos$b) {
    gp$heights[guide.pos$t] <- unit(0, "pt")
  }
  # guide.grob <- gp[["layout"]][guide.grob, ]
  # if(guide.grob[["l"]] == guide.grob[["r"]]){
  #   gp <- gtable_squash_cols(gp, cols = guide.grob[["l"]])
  # }
  # if(guide.grob[["t"]] == guide.grob[["b"]]){
  #   gp <- gtable_squash_rows(gp, rows = guide.grob[["t"]])
  # }
  #gp <- gtable_remove_grobs(gp, "guide-box")
  gp$grobs[[guide.idx]]=zeroGrob()
  return(gp)
}


##################################################################
### Generate Across and Within Establishment Summaries ##########
#################################################################
genprops=function(nestab,shapev,scalev,
                  nreps=4,iter=1,
                  avg_per_estab=c(15,15,15,165000),
                  lambda=NULL,theta=NULL,
                  lambda_funct=NULL,theta_funct=NULL,
                  wagemin=1,
                  experiment=F,
                  return_reduced=T){
  if(is.null(lambda)==T){
    lambda=shapev
    lambda_funct="identity"
  }
  if(is.null(theta)==T){
    theta=scalev
    theta_funct="identity"
  }
  all_iter_id=paste0("d",nestab,
                     '_lambda',round(lambda,2),"_lambdafunct-",lambda_funct,
                     '_theta',round(theta,2),'_thetafunct-',theta_funct,
                     '_nreps',nreps,'_iter',iter)
  if((experiment==T)&(nreps>1)){
    priors=rgamma(nestab,shape=shapev,scale=scalev)
    genpropsdf=dplyr::bind_rows(lapply(avg_per_estab,
                                       function(avg) as.data.frame(
                                         MCMCprecision::rdirichlet(1,1e-6+avg+priors))))
  }else{
  genpropsdf=as.data.frame(MCMCprecision::rdirichlet(nreps,
                                                     1e-6+rgamma(nestab,shape=shapev,scale=scalev)))
  }
  gencountdf=genpropsdf
  if(nreps>1){
    if(nreps==4){
      for(r in seq(1,3)){
        gencountdf[r,]=gencountdf[r,]*(avg_per_estab[r]*nestab)
      }
      gencountdf[4,]=wagemin+(gencountdf[4,]*(avg_per_estab[4]-wagemin)*nestab)
    }else{
    for(r in seq(1,nreps)){
      gencountdf[r,]=gencountdf[r,]*(avg_per_estab[r]*nestab)
    }
    }
  }else{
    if(length(avg_per_estab)==1){
      gencountdf=gencountdf*(avg_per_estab*nestab)
    }else{
      gencountdf=gencountdf*(avg_per_estab[1]*nestab)
    }
    
  }
  if(nreps>1){
    genpropsdf=genpropsdf[seq(1,nreps),]
    gencountdf=gencountdf[seq(1,nreps),]
  }
  
  genprops_add_stats=function(df,zeros=NULL,
                              numreps=nreps,shapeval=shapev,scaleval=scalev,
                              i_iteration=iter,numestab=nestab){
    df$spread_props=df$maxval_props-df$minval_props
    df$spread_counts=df$maxval_counts-df$minval_counts
    
    if(!is.null(zeros)){
      df$count2digzeros=zeros
      df$prop2digzeros=zeros/numestab
    }
    if("shape" %in% colnames(df)){
      return(df)
    }else{
      df$shape=shapeval
      df$scale=scaleval
      df$nestab=numestab
      df$nreps=numreps
      df$iter=i_iteration
      return(df)
    }
  }
  
  ac_rzeros=rowSums(round(gencountdf,0)==0)
  
  if(sum(is.na(genpropsdf))>0){
    print(all_iter_id)
    print(head(genpropsdf))
    print(head(gencountdf))
  }
  acrossestab_summary_props=apply(genpropsdf,1,function(x)c(quantile(x),mean(x),sd(x),sqrt(var(x)/nestab)))
  acrossestab_summary_counts=apply(gencountdf,1,function(x)c(quantile(x),mean(x),sd(x),sqrt(var(x)/nestab)))
  rownames(acrossestab_summary_props)=paste0(c("minval","Q1val","medianval","Q3val","maxval","mean","sd","se"),"_props")
  rownames(acrossestab_summary_counts)=paste0(c("minval","Q1val","medianval","Q3val","maxval","mean","sd","se"),"_counts")
  acrossestab_summary=as.data.frame(t(dplyr::bind_rows(list(as.data.frame(acrossestab_summary_props),
                                                            as.data.frame(acrossestab_summary_counts)))))
  acrossestab_summary$repid=seq(1,nreps)
  acrossestab_summary=genprops_add_stats(df=acrossestab_summary,zeros=ac_rzeros)
  acrossestab_summary$iterid=all_iter_id
  acrossestab_summary$lambda=lambda
  acrossestab_summary$theta=theta
  acrossestab_summary$lambda_funct=lambda_funct
  acrossestab_summary$theta_funct=theta_funct
  
  if(nreps>1){
    wi_czeros=colSums(round(gencountdf,0)==0)
    withinestab_summary_props=t(apply(genpropsdf,2,function(x)c(quantile(x),mean(x),sd(x))))
    colnames(withinestab_summary_props)=paste0(c("minval","Q1val","medianval","Q3val","maxval","mean","sd"),"_props")
    if(nreps>3){
      withinestab_summary_counts=t(apply(gencountdf,2,function(x)c(quantile(x),mean(x),sd(x),sd(x[1:3]))))
    colnames(withinestab_summary_counts)=paste0(c("minval","Q1val","medianval","Q3val","maxval","mean","sd","sd_emp"),"_counts")
    }else{
    withinestab_summary_counts=t(apply(gencountdf,2,function(x)c(quantile(x),mean(x),sd(x))))
    colnames(withinestab_summary_counts)=paste0(c("minval","Q1val","medianval","Q3val","maxval","mean","sd"),"_counts")
    }
    withinestab_summary=dplyr::bind_cols(list(as.data.frame(withinestab_summary_counts),
                                              as.data.frame(withinestab_summary_props)))
    rownames(withinestab_summary)=paste0("estab",seq(1,nestab))
    withinestab_summary=as.data.frame(withinestab_summary)
    withinestab_summary$estabid=seq(1,nestab)
    withinestab_summary=genprops_add_stats(df=withinestab_summary,zeros=wi_czeros)
    withinestab_summary$prop2digzeros=wi_czeros/nreps
    withinestab_summary$count2digzeros=wi_czeros
    withinestab_summary$iterid=all_iter_id
    withinestab_summary$lambda=lambda
    withinestab_summary$theta=theta
    withinestab_summary$lambda_funct=lambda_funct
    withinestab_summary$theta_funct=theta_funct
    if(return_reduced==T){
      acrossestab_summary$scope="across"
      withinestab_summary$scope="within"
      if(nreps==4){
        return(data.table::rbindlist(
          list(acrossestab_summary[acrossestab_summary$repid%in%c(1,4),],
               withinestab_summary[withinestab_summary$estabid==1,]),fill=TRUE))
      }else{
        return(data.table::rbinlist(list(acrossestab_summary[acrossestab_summary$repid==1,],
                                         withinestab_summary[withinestab_summary$estabid==1,]),fill=T))
        
      }
          }else{
      return(list(acrossestab_summary,withinestab_summary))
    }
    
  }else{
    if(return_reduced==T){ 
      acrossestab_summary$scope="across"
    if(nreps==4){
      return(acrossestab_summary[acrossestab_summary$repid%in%c(1,4),])
      
    }else{
      return(acrossestab_summary[acrossestab_summary$repid==1,])
      
    }
    }else{
    return(acrossestab_summary)
  }
  }
}

##################################################################
### Add Label for use in plotting to data #######
##################################################################
add_labels_data=function(withinestab,nestabs_list,nestab_labs_list,theta_list,lambda_list){
  #all_na_col=sapply(withinestab,function(x)all(is.na(x)))
  #withinestab=withinestab[,!all_na_col]
  col_nestab_lab_sfx=nestab_labs_list[sapply(withinestab$nestab,function(x)which(nestabs_list==x))]
  withinestab$nestab_lab=factor(paste0("Establishment Count=",withinestab$nestab," (",col_nestab_lab_sfx,")"),
                                levels=paste0("Establishment Count=",nestabs_list," (",nestab_labs_list,")"))
  
  withinestab$theta_lab=factor(paste0("theta==",withinestab$theta),
                               levels=paste0("theta==",theta_list))
  withinestab$lambda_lab=factor(paste0("lambda==",withinestab$lambda),
                                levels=paste0("lambda==",lambda_list))
  return(withinestab)
}


####################################################################################################################################
###### Generalized Plotting Function ###############
########################################################################################
handle_addtheme=function(add.theme,plt){
  if((is.list(add.theme))&(length(add.theme)>1)){
    for(item in add.theme){
      plt=plt+item
    }
  }else{
    plt=plt+add.theme
  }
  return(plt)
}


plot_gamma_smooth <- function(data,
                              xvar,
                              yvar,
                              colvar,
                              across=T,
                              facetvar   = "nestab_lab",
                              free_y      = FALSE,
                              nrow_legend = 3,
                              savefolder   = NULL,
                              fname_stem=NULL,
                              width       = 7.7,
                              height      = 3,
                              units       = "in",
                              shape_transform="identity",
                              scale_transform="identity",
                              include_points=F,
                              addtheme=NULL) {
### 
  # Label for y variable
  ylab_lookup <- c(
    se_counts      = "Count Standard Error",
    se_props       = "Proportion Standard Error",
    prop2digzeros  = "Proportion of Establishment=0",
    count2digzeros = "Count of Establishment=0",
    spread_props=" Range of Proportions",
    sd_props="Proportion Standard Deviation",
    sd_counts="Count Standard Deviation",
    sd_emp_counts="Employment Count Standard Deviation"
  )
  yvarlabel=ylab_lookup[yvar]
  if (is.na(yvarlabel)){yvarlabel = yvar}   # fallback
  titleparts=c(", for Prior Gamma(Shape=",", Scale=")
  acrosstitle=paste0(") ",ifelse(across==T,"Across","Within")," Establishments.")
  if(shape_transform=="identity"){
    shape_transform=rlang::sym("lambda")
  }
  if(scale_transform=="identity"){
    scale_transform=rlang::sym("theta")
  }
  titleexpr=bquote(.(yvarlabel)~.(titleparts[1])~.(shape_transform)~.(titleparts[2])~.(scale_transform)~.(acrosstitle))
  
  if(scale_transform=="over_lambda"){
    titleexpr=bquote(.(yvarlabel)~.(titleparts[1])~.(shape_transform)~.(titleparts[2])~frac(theta,lambda)~.(acrosstitle))
    
  }
  if(shape_transform=="over_d"){
    titleexpr=bquote(.(yvarlabel)~.(titleparts[1])~frac(lambda,d)~.(titleparts[2])~theta~.(acrosstitle))
    
  }
  if((shape_transform=="over_d")&(scale_transform=="over_lambda")){
    titleexpr=bquote(.(yvarlabel)~.(titleparts[1])~frac(lambda,d)~.(titleparts[2])~frac(theta,lambda)~.(acrosstitle))
    
  }
  
  if((shape_transform=="over_d")&(scale_transform=="over_lambda_over_d")){
    titleexpr=bquote(.(yvarlabel)~.(titleparts[1])~frac(lambda,d)~.(titleparts[2])~d*frac(theta,lambda)~.(acrosstitle))
    
  }
  
    if (is.null(titleexpr)){titleexpr <- paste(ifelse(across==T,"Across Establishment ","Within Establishment "), yvar)}
  
  lev =levels(data[[colvar]])
  data[[colvar]]=factor(as.character(data[[colvar]]),levels=lev[lev %in% unique(as.character(data[[colvar]]))])
  lev =levels(data[[colvar]])
  if (is.null(lev)){lev=sort(unique(data[[colvar]]))}   # character fallback
  parsedlabs = parse(text = lev)   # theta==1 renders correctly as θ=1
  
  nlev=length(lev)
  linecols=RColorBrewer::brewer.pal(max(nlev,3),"Dark2")[seq_len(nlev)]
  fillcols=RColorBrewer::brewer.pal(max(nlev,3),"Set2")[seq_len(nlev)]
  facetscale=ifelse(free_y==T,"free_y","fixed")
  colexpr=ifelse(colvar=="theta_lab",expression("*theta*"),
                 ifelse(colvar=="lambda_lab",expression("*lambda*"),colvar))
  
  xexpr=ifelse(xvar=="theta",rlang::sym("theta"),
                 ifelse(xvar=="lambda",rlang::sym("lambda"),xvar))
  
  p <- ggplot(data,
              aes(x     = .data[[xvar]],
                  y     = .data[[yvar]],
                  color = .data[[colvar]],
                  fill  = .data[[colvar]],
                  group=.data[[colvar]]))
  if(include_points==T){
    p<-p+geom_point(alpha=0.4,size=0.4,aes(fill=.data[[colvar]],color=.data[[colvar]],group=.data[[colvar]]),
                    position=position_jitter(width=1.5))
    #position_jitterdodge(dodge.width=1,#0.5,jitter.width=1.5))#0.1))
  }
  p<-p+
    geom_smooth(se = TRUE,method = "loess") +
    labs(x = xexpr, y = yvarlabel) +
    ggtitle(titleexpr)+
    scale_color_brewer(
      name    = colexpr,
      palette = "Dark2",
      labels  = parsedlabs
    ) +
    scale_fill_brewer(
      name    = colexpr,        # must match scale_color_brewer name
      palette = "Set2",
      labels  = parsedlabs
    ) +
    guides(
      color = guide_legend(
        title = NULL,
        nrow           = nrow_legend,
        byrow=TRUE,
        override.aes   = list(
          color     = linecols,
          fill      = fillcols,
          linewidth = 1
        )
      ),
      fill = "none"              # suppress duplicate fill legend
    ) +
    presettheme +
    facet_wrap(reformulate(facetvar), scales = facetscale)
  if(!is.null(addtheme)){
    p=handle_addtheme(addtheme,p)
  }
  
  pshifted=shift_legend(p)
  grid.draw(pshifted)
  
  # ── Optionally save ───────────────────────────────────────────────────────
  if ((!is.null(savefolder))|(!is.null(fname_stem))) {
    fname=paste0(ifelse(across==T,"acrossestab","withinestab"),"_",yvar,"_x_",xvar,"_",fname_stem,".pdf")
    if((savefolder=="")|(endsWith("/",savefolder))){
      ggsave(paste0(savefolder,fname), plot = pshifted, width = width, height = height, units = units)
      message("Saved: ",paste0(savefolder,fname))
    }else{
      ggsave(paste(savefolder,fname,sep="/"), plot = pshifted, width = width, height = height, units = units)
      message("Saved: ",paste(savefolder,fname,sep="/"))
    }
    
  }
  
  invisible(pshifted)   # return the ggplot object quietly
}

####################################################################################################################################
####################################################################################################################################


handle_addtheme=function(add.theme,plt){
  if((is.list(add.theme))&(length(add.theme)>1)){
      for(item in add.theme){
      plt=plt+item
    }
  }else{
    plt=plt+add.theme
  }
  return(plt)
}


####################################################################################################################################
####################################################################################################################################
####################################################################################################################################

  

#Constant Parameters
nestabs_list=c(3,6,15,20,71)#,244)
nestab_labs_list=c("Q1","median","Q3","mean","95-percentile")#,"99-percentile")
niters=10
avg_perestnum=c(7,7,7,56)
nreps=4
presettheme=theme_bw(base_size = 8)+theme(#axis.text.x = element_blank(),
  plot.title = element_text(size=7)#,
  #axis.title.x = element_blank(),
)
ae_free_y_lookup=c(se_counts= FALSE,se_props= FALSE,
                   prop2digzeros  = FALSE,count2digzeros = FALSE,
                   spread_props=FALSE,
                   sd_props=FALSE,sd_counts=FALSE,
                   skew_props=FALSE,kurtosis_props=FALSE,
                   skew_counts=FALSE,kurtosis_counts=FALSE)
we_free_y_lookup=c(se_counts= FALSE,se_props= FALSE,
                   prop2digzeros  = FALSE,count2digzeros = FALSE,
                   spread_props=FALSE,
                   sd_props=FALSE,sd_counts=FALSE,
                   skew_props=FALSE,kurtosis_props=FALSE,
                   skew_emp_counts=FALSE,kurtosis_emp_counts=FALSE,
                   sd_emp_counts=FALSE)



###scale dataframes
niters=100
lambda_list=c(1.5,2.5,5,1.7,10,15,20,25,30,40,50,75,100,125)#c(seq(1.5,9,0.5),seq(10,30,4),seq(40,70,10))
theta_list=c(0.1,0.5,1,2,4,6,8,10,15,20,30,40,50,75,100,150,200,250)
combination=base::expand.grid(nestabs_list,lambda_list,theta_list)

combination[,4]=combination[,2]/combination[,1]
combination[,5]=combination[,3]
colnames(combination)=c("nestab","lambda","theta","shape","scale")


withinestab_og=list()
acrossestab_og=list()
set.seed(9)
#cl=makeCluster(detectCores()-1)
#clusterExport()
og_df_list=list()
for(i in seq(1,niters)){
  temp=parallel::mclapply(seq(1,nrow(combination)),
                          function(rw)genprops(combination[rw,1],combination[rw,4],combination[rw,5],
                                               nreps=nreps,iter=i,avg_per_estab=avg_perestnum,
                                               lambda=combination[rw,2],theta=combination[rw,3],
                                               lambda_funct="over_d",theta_funct="identity"))
  og_df_list[[i]]=data.table::rbindlist(temp)
}
fulldf_og=data.table::rbindlist(og_df_list)
withinestab_og=fulldf_og[fulldf_og$scope=="within",]
acrossestab_og=fulldf_og[fulldf_og$scope=="across"]

# for(i in seq(1,niters)){
#   temp=apply(combination,1,
#              function(x)genprops(x[1],x[4],x[5],nreps=nreps,iter=i,avg_per_estab=avg_perestnum,
#                                  lambda=x[2],theta=x[3],
#                                  lambda_funct="over_d",theta_funct="identity"))
#   acrossestab_og[[i]]=data.table::::rbindlist(lapply(temp,"[[",1))
#   withinestab_og[[i]]=data.table::::rbindlist(lapply(temp,"[[",2))
# }
# withinestab_og=data.table::::rbindlist(withinestab_og)
# acrossestab_og=data.table::::rbindlist(acrossestab_og)

withinestab_og=add_labels_data(withinestab_og,nestabs_list,nestab_labs_list,theta_list,lambda_list)
acrossestab_og=add_labels_data(acrossestab_og,nestabs_list,nestab_labs_list,theta_list,lambda_list)

theta_discrete= c(0.1,0.5,1,2,5,10,15,20)
theta_range=c(0.75,51)
lambda_discrete=c(1.5,2,5,10,20,30,50,75,100)
lambda_range=c(0.75,101)
#theta_list=c(0.1,0.5,1,5,10,15,20,30,40,50,75,100)
ae_plot_theta_og=acrossestab_og[(acrossestab_og$theta %in% theta_discrete)&
                                  (acrossestab_og$lambda<lambda_range[2])&
                                  (acrossestab_og$lambda>lambda_range[1]),]
we_plot_theta_og=withinestab_og[(withinestab_og$theta %in% theta_discrete)&
                                  (withinestab_og$lambda<lambda_range[2])&
                                  (withinestab_og$lambda>lambda_range[1]),]
#lambda_list=c(1.5,5,10,15,20,30,40,50,75,100)
ae_plot_lambda_og=acrossestab_og[(acrossestab_og$lambda %in% lambda_discrete)&
                                   (acrossestab_og$theta<theta_range[2])&
                                   (acrossestab_og$theta>theta_range[1]),]
we_plot_lambda_og=withinestab_og[(withinestab_og$lambda %in% lambda_discrete)&
                                   (withinestab_og$theta<theta_range[2])&
                                   (withinestab_og$theta>theta_range[1]),]

og_fstem="shape_lambda_over_d_scale_theta_wagemin_iter100"
dir.create(og_fstem)


for(vstr in c("prop2digzeros","spread_props","sd_counts","sd_props")){
  plot_gamma_smooth(ae_plot_theta_og,"lambda",vstr,
                    colvar="theta_lab",across=T,free_y=ae_free_y_lookup[vstr],
                    savefolder = og_fstem,
                    fname_stem=og_fstem,
                    shape_transform = "over_d",
                    scale_transform="identity",
                    include_points=F)#ifelse(vstr%in%c("prop2digzeros","spread_props"),T,F))
  plot_gamma_smooth(ae_plot_lambda_og,"theta",vstr,
                    colvar="lambda_lab",across=T,free_y=ae_free_y_lookup[vstr],
                    savefolder = og_fstem,
                    fname_stem=og_fstem,
                    shape_transform = "over_d",
                    scale_transform="identity",
                    include_points=F)#ifelse(vstr%in%c("prop2digzeros","spread_props"),T,F))
}

for(vstr in c("sd_props","spread_props","count2digzeros","sd_emp_counts")){
  plot_gamma_smooth(we_plot_theta_og,"lambda",vstr,
                    colvar="theta_lab",across=F,free_y=we_free_y_lookup[vstr],
                    savefolder = og_fstem,
                    fname_stem=og_fstem,
                    shape_transform = "over_d",
                    scale_transform="identity",
                    include_points=F)#ifelse(vstr%in%c("prop2digzeros","count2digzeros","spread_props"),T,F))
  plot_gamma_smooth(we_plot_lambda_og,"theta",vstr,
                    colvar="lambda_lab",across=F,free_y=we_free_y_lookup[vstr],
                    savefolder = og_fstem,
                    fname_stem=og_fstem,
                    shape_transform = "over_d",
                    scale_transform="identity",
                    include_points=F)#ifelse(vstr%in%c("prop2digzeros","count2digzeros","spread_props"),T,F))
}


###scale dataframes
lambda_list=c(2,5,10,15,20,30,40,50,75,100)#c(seq(1.5,9,0.5),seq(10,30,4),seq(40,70,10))
theta_list=c(1,5,10,15,20,30,40,50,75,100,150,200,250)
combination=base::expand.grid(nestabs_list,lambda_list,theta_list)

combination[,4]=combination[,2]/combination[,1]
combination[,5]=combination[,3]/combination[,2]
colnames(combination)=c("nestab","lambda","theta","shape","scale")


withinestab_p2=data.frame()
acrossestab_p2=data.frame()
set.seed(9)
for(i in seq(1,niters)){
  temp=apply(combination,1,
             function(x)genprops(x[1],x[4],x[5],nreps=nreps,iter=i,avg_per_estab=avg_perestnum,
                                 lambda=x[2],theta=x[3],
                                 lambda_funct="over_d",theta_funct="over_lambda"))
  acrossestab_i=dplyr::bind_rows(lapply(temp,"[[",1))
  withinestab_i=dplyr::bind_rows(lapply(temp,"[[",2))
  withinestab_p2=dplyr::bind_rows(list(withinestab_p2,withinestab_i))
  acrossestab_p2=dplyr::bind_rows(list(acrossestab_p2,acrossestab_i))
}

withinestab_p2=add_labels_data(withinestab_p2,nestabs_list,nestab_labs_list,theta_list,lambda_list)
acrossestab_p2=add_labels_data(acrossestab_p2,nestabs_list,nestab_labs_list,theta_list,lambda_list)

#theta_list=c(0.1,0.5,1,5,10,15,20,30,40,50,75,100)
ae_plot_theta_p2=acrossestab_p2[(acrossestab_p2$repid==1)&
                                  (acrossestab_p2$theta %in% c(10,30,50,75,100,150,200,250))&
                                  (acrossestab_p2$lambda<101),]
we_plot_theta_p2=withinestab_p2[(withinestab_p2$estabid==1)&
                                  (withinestab_p2$theta %in% c(10,30,50,75,100,150,200,250))&
                                  (withinestab_p2$lambda<101),]
#lambda_list=c(1.5,5,10,15,20,30,40,50,75,100)
ae_plot_lambda_p2=acrossestab_p2[(acrossestab_p2$repid==1)&
                                   (acrossestab_p2$lambda %in% c(1.5,2,5,10,20,30,50,75,100))&
                                   (acrossestab_p2$theta<101)&(acrossestab_p2$theta>4),]
we_plot_lambda_p2=withinestab_p2[(withinestab_p2$estabid==1)&
                                   (withinestab_p2$lambda %in% c(1.5,2,5,10,20,30,50,75,100))&
                                   (withinestab_p2$theta<101)&(withinestab_p2$theta>4),]

p2_fstem="shape_lambda_over_d_scale_theta_over_lambda"
dir.create(p2_fstem)

for(vstr in c("se_counts","prop2digzeros","spread_props")){
  plot_gamma_smooth(ae_plot_theta_p2,"lambda",vstr,
                    colvar="theta_lab",across=T,free_y=ae_free_y_lookup[vstr],
                    savefolder = p2_fstem,
                    fname_stem=p2_fstem,
                    shape_transform = "over_d",
                    scale_transform="over_lambda",
                    include_points=ifelse(vstr%in%c("prop2digzeros","spread_props"),T,F))
  plot_gamma_smooth(ae_plot_lambda_p2,"theta",vstr,
                    colvar="lambda_lab",across=T,free_y=ae_free_y_lookup[vstr],
                    savefolder = p2_fstem,
                    fname_stem=p2_fstem,
                    shape_transform = "over_d",
                    scale_transform="over_lambda",
                    include_points=ifelse(vstr%in%c("prop2digzeros","spread_props"),T,F))
}

for(vstr in c("prop2digzeros","se_props","spread_props","count2digzeros")){
  plot_gamma_smooth(we_plot_theta_p2,"lambda",vstr,
                    colvar="theta_lab",across=F,free_y=we_free_y_lookup[vstr],
                    savefolder = p2_fstem,
                    fname_stem=p2_fstem,
                    shape_transform = "over_d",
                    scale_transform="over_lambda")
  plot_gamma_smooth(we_plot_lambda_p2,"theta",vstr,
                    colvar="lambda_lab",across=F,free_y=we_free_y_lookup[vstr],
                    savefolder = p2_fstem,
                    fname_stem=p2_fstem,
                    shape_transform = "over_d",
                    scale_transform="over_lambda")
}






plot_gamma_smooth(ae_plot_theta_og,"lambda","se_counts",
                  colvar="theta_lab",across=T,free_y=F,
                  savefolder = "",
                  fname_stem=og_fstem,
                  shape_transform = "over_d")
plot_gamma_smooth(ae_plot_theta_og,"lambda","prop2digzeros",
                  colvar="theta_lab",across=T,free_y=F,
                  savefolder = "",
                  fname_stem=og_fstem,
                  shape_transform = "over_d")
plot_gamma_smooth(ae_plot_lambda_og,"theta","se_counts",
                  colvar="lambda_lab",across=T,free_y=F,
                  fname_stem=og_fstem,
                  shape_transform = "over_d")
plot_gamma_smooth(ae_plot_lambda_og,"theta","prop2digzeros",
                  colvar="lambda_lab",across=T,free_y=F,
                  savefolder = "",
                  fname_stem=og_fstem,
                  shape_transform = "over_d")

plot_gamma_smooth(we_plot_theta_og,"lambda","se_props",
                  colvar="theta_lab",across=F,free_y=F,
                  savefolder = "",
                  fname_stem=og_fstem,
                  shape_transform = "over_d")
plot_gamma_smooth(we_plot_theta_og,"lambda","prop2digzeros",
                  colvar="theta_lab",across=F,free_y=F,
                  savefolder = "",
                  fname_stem=og_fstem,
                  shape_transform = "over_d")
plot_gamma_smooth(we_plot_lambda_og,"theta","se_props",
                  colvar="lambda_lab",across=F,free_y=F,
                  savefolder = "",
                  fname_stem=og_fstem,
                  shape_transform = "over_d")
plot_gamma_smooth(we_plot_lambda_og,"theta","prop2digzeros",
                  colvar="lambda_lab",across=F,free_y=F,
                  savefolder = "",
                  fname_stem=og_fstem,
                  shape_transform = "over_d")



###scale dataframes
lambda_list=c(2,5,10,15,20,30,40,50,75,100)#c(seq(1.5,9,0.5),seq(10,30,4),seq(40,70,10))
theta_list=c(1,5,10,15,20,30,40,50,75,100)
combination=base::expand.grid(nestabs_list,lambda_list,theta_list)

combination[,4]=combination[,2]/combination[,1]
combination[,5]=combination[,3]/combination[,4]
colnames(combination)=c("nestab","lambda","theta","shape","scale")


withinestab_p2=data.frame()
acrossestab_p2=data.frame()
set.seed(9)
for(i in seq(1,niters)){
  temp=lapply(combination,
             function(x)genprops(x[1],x[4],x[5],nreps=nreps,iter=i,avg_per_estab=avg_perestnum,
                                 lambda=x[2],theta=x[3],
                                 lambda_funct="over_d",theta_funct="over_lambda"))
  acrossestab_i=dplyr::bind_rows(lapply(temp,"[[",1))
  withinestab_i=dplyr::bind_rows(lapply(temp,"[[",2))
  withinestab_p2=dplyr::bind_rows(list(withinestab_p2,withinestab_i))
  acrossestab_p2=dplyr::bind_rows(list(acrossestab_p2,acrossestab_i))
}

withinestab_p2=add_labels_data(withinestab_p2,nestabs_list,nestab_labs_list,theta_list,lambda_list)
acrossestab_p2=add_labels_data(acrossestab_p2,nestabs_list,nestab_labs_list,theta_list,lambda_list)

#theta_list=c(0.1,0.5,1,5,10,15,20,30,40,50,75,100)
ae_plot_theta_p2=acrossestab_p2[(acrossestab_p2$repid==1)&(acrossestab_p2$theta %in% c(0.5,1,5,10,20,50,100))&(acrossestab_p2$lambda<101),]
we_plot_theta_p2=withinestab_p2[(withinestab_p2$estabid==1)&(withinestab_p2$theta %in% c(0.5,1,5,10,20,50,100))&(withinestab_p2$lambda<101),]
#lambda_list=c(1.5,5,10,15,20,30,40,50,75,100)
ae_plot_lambda_p2=acrossestab_p2[(acrossestab_p2$repid==1)&(acrossestab_p2$lambda %in% c(1.5,2,5,10,20,50,75))&(acrossestab_p2$theta<76)&(acrossestab_p2$theta>4),]
we_plot_lambda_p2=withinestab_p2[(withinestab_p2$estabid==1)&(withinestab_p2$lambda %in% c(1.5,2,5,10,20,50,75))&(withinestab_p2$theta<76)&(withinestab_p2$theta>4),]

p2_fstem="shape_lambda_over_d_scale_theta_over_lambda_over_d"
plot_gamma_smooth(ae_plot_theta_p2,"lambda","se_counts",
                  colvar="theta_lab",across=T,free_y=F,
                  savefolder = "",
                  fname_stem=p2_fstem,
                  shape_transform = "over_d",
                  scale_transform = "over_lambda_over_d")
plot_gamma_smooth(ae_plot_theta_p2,"lambda","prop2digzeros",
                  colvar="theta_lab",across=T,free_y=F,
                  savefolder = "",
                  fname_stem=p2_fstem,
                  shape_transform = "over_d",
                  scale_transform = "over_lambda_over_d")
plot_gamma_smooth(ae_plot_lambda_p2,"theta","se_counts",
                  colvar="lambda_lab",across=T,free_y=F,
                  fname_stem=p2_fstem,
                  shape_transform = "over_d",
                  scale_transform = "over_lambda_over_d")
plot_gamma_smooth(ae_plot_lambda_p2,"theta","prop2digzeros",
                  colvar="lambda_lab",across=T,free_y=F,
                  savefolder = "",
                  fname_stem=p2_fstem,
                  shape_transform = "over_d",
                  scale_transform = "over_lambda_over_d")

plot_gamma_smooth(we_plot_theta_p2,"lambda","se_props",
                  colvar="theta_lab",across=F,free_y=F,
                  savefolder = "",
                  fname_stem=p2_fstem,
                  shape_transform = "over_d",
                  scale_transform = "over_lambda_over_d")
plot_gamma_smooth(we_plot_theta_p2,"lambda","prop2digzeros",
                  colvar="theta_lab",across=F,free_y=F,
                  savefolder = "",
                  fname_stem=p2_fstem,
                  shape_transform = "over_d",
                  scale_transform = "over_lambda_over_d")
plot_gamma_smooth(we_plot_lambda_p2,"theta","se_props",
                  colvar="lambda_lab",across=F,free_y=F,
                  savefolder = "",
                  fname_stem=p2_fstem,
                  shape_transform = "over_d",
                  scale_transform = "over_lambda_over_d")
plot_gamma_smooth(we_plot_lambda_p2,"theta","prop2digzeros",
                  colvar="lambda_lab",across=F,free_y=F,
                  savefolder = "",
                  fname_stem=p2_fstem,
                  shape_transform = "over_d",
                  scale_transform = "over_lambda_over_d")


############################################
secount_across=bquote("Standard Error of Counts, for Prior Gamma(Shape="*lambda*" Scale="*theta*"/"*lambda*") Across Establishments.")
seprop_across=bquote("Proportion Standard Error, for Prior Gamma(Shape="*lambda*" Scale="*theta*"/"*lambda*") Across Establishments.")
propzero_across=bquote("Proportion of Zeros, for Prior Gamma(Shape="*lambda*" Scale="*theta*"/"*lambda*") Across Establishments.")
countzero_across=bquote("Count of Zeros, for Prior Gamma(Shape="*lambda*" Scale="*theta*"/"*lambda*") Across Establishments.")

secount_within=bquote("Standard Error of Counts, for Prior Gamma(Shape="*lambda*" Scale="*theta*"/"*lambda*") Within Establishments.")
seprop_within=bquote("Proportion Standard Error, for Prior Gamma(Shape="*lambda*" Scale="*theta*"/"*lambda*") Within Establishments.")
propzero_within=bquote("Proportion of Zeros, for Prior Gamma(Shape="*lambda*" Scale="*theta*"/"*lambda*") Within Establishments.")
countzero_within=bquote("Count of Zeros, for Prior Gamma(Shape="*lambda*" Scale="*theta*"/"*lambda*") Within Establishments.")

ae_plot_theta_less=ae_plot_theta[ae_plot_theta$nestab<200,]
plt1=ggplot(ae_plot_theta_less,aes(x=lambda,y=se_counts,col=theta_lab,fill=theta_lab))+
  #geom_point(alpha=0.1)+
  geom_smooth(se=T)+
  labs(x=expression(""*lambda*""),y="Standard Error")+  
  ggtitle(secount_across)+
  scale_color_brewer(name=expression(theta),palette="Dark2",labels=parse(text=levels(ae_plot_theta_less$theta_lab)))+
  scale_fill_brewer(name=expression(theta),palette="Set2",labels=parse(text=levels(ae_plot_theta_less$theta_lab)))+
  presettheme+
  facet_wrap(.~nestab_lab)

plt1_new=plt1+guides(col=guide_legend(title.position="top",nrow=3),
                     fill=guide_legend(title.position="top",nrow=3))
grid.draw(shift_legend(plt1_new))
ggsave(paste(plotfolder,"/acrossestab_se_counts_x_lambda_identity_over_lambda.pdf"),width=7.7,height=3,units='in')


ggplot(ae_plot_theta,aes(x=lambda,y=prop2digzeros,col=theta_lab,fill=theta_lab))+
  #geom_point(alpha=0.1)+
  geom_smooth(se=T)+
  labs(x=expression(""*lambda*""),col=expression(""*theta*""),fill=expression(""*theta*""),y="Proportion of Establishment=0")+
  ggtitle(propzero_across)+
  scale_color_brewer(palette="Dark2")+
  scale_fill_brewer(palette="Set2")+
  presettheme+
  facet_wrap(.~nestab_lab,scale="free_y")
ggsave(paste(plotfolder,"/acrossestab_propzeros_x_lambda_identity_over_lambda.pdf"),width=7.7,height=3,units='in')


ggplot(ae_plot_lambda,aes(x=theta,y=se_counts,col=lambda_lab,fill=lambda_lab))+
  #geom_point(alpha=0.1)+
  geom_smooth(se=T)+
  labs(x=expression(""*theta*""),col=expression(""*lambda*""),fill=expression(""*lambda*""),y="Count Standard Error")+
  ggtitle(secount_across)+
  scale_color_brewer(palette="Dark2")+
  scale_fill_brewer(palette="Set2")+
  presettheme+
  facet_wrap(.~nestab_lab,scale="free_y")
ggsave(paste(plotfolder,"/acrossestab_se_counts_x_lambda_identity_over_lambda.pdf"),width=7.7,height=3,units='in')


ggplot(ae_plot_lambda,aes(x=theta,y=se_props,col=lambda_lab))+
  #geom_point(alpha=0.1)+
  geom_smooth(se=T)+
  labs(x=expression(""*theta*""),col=expression(""*lambda*""),y="Proportion Standard Error")+
  ggtitle(seprop_across)+
  #scale_color_brewer(palette="Dark2")+
  presettheme+
  facet_wrap(.~nestab_lab,scale="free_y")

ggplot(ae_plot_lambda,aes(x=theta,y=prop2digzeros,col=lambda_lab,fill=lambda_lab))+
  #geom_point(alpha=0.1)+
  geom_smooth(se=T)+
  labs(x=expression(""*theta*""),col=expression(""*lambda*""),fill=expression(""*lambda*""),y="Proportion of Establishment=0")+
  ggtitle(propzero_across)+
  scale_color_brewer(palette="Dark2")+
  scale_fill_brewer(palette="Set2")+
  presettheme+
  facet_wrap(.~nestab_lab)#,scale="free_y")

ggplot(ae_plot_lambda,aes(x=theta,y=count2digzeros,col=lambda_lab,fill=lambda_lab))+
  #geom_point(alpha=0.1)+
  geom_smooth(se=T)+
  labs(x=expression(""*theta*""),col=expression(""*lambda*""),fill=expression(""*lambda*""),y="Count of Establishment=0")+
  ggtitle(countzero_across)+
  scale_color_brewer(palette="Dark2")+
  scale_fill_brewer(palette="Set2")+
  presettheme+
  facet_wrap(.~nestab_lab,scale="free_y")

### Repeat for WITHINESTAB
ggplot(we_plot_theta,aes(x=lambda,y=se_props,col=theta_lab,fill=theta_lab))+
  #geom_point(alpha=0.1)+
  geom_smooth(se=T)+
  labs(x=expression(""*lambda*""),col=expression(""*theta*""),fill=expression(""*theta*""),y="Standard Error")+
  ggtitle(seprop_within)+
  scale_color_brewer(palette="Dark2")+
  scale_fill_brewer(palette="Set2")+
  presettheme+
  facet_wrap(.~nestab_lab)

ggplot(we_plot_theta,aes(x=lambda,y=se_counts,col=theta_lab,fill=theta_lab))+
  #geom_point(alpha=0.1)+
  geom_smooth(se=T)+
  labs(x=expression(""*lambda*""),col=expression(""*theta*""),fill=expression(""*theta*""),y="Standard Error")+
  ggtitle(secount_within)+
  scale_color_brewer(palette="Dark2")+
  scale_fill_brewer(palette="Set2")+
  presettheme+
  facet_wrap(.~nestab_lab)


ggplot(we_plot_theta,aes(x=lambda,y=count2digzeros,col=theta_lab,fill=theta_lab))+
  #geom_point(alpha=0.1)+
  geom_smooth(se=T)+
  labs(x=expression(""*lambda*""),col=expression(""*theta*""),fill=expression(""*theta*""),y="Count of Variables=0")+
  ggtitle(countzero_within)+
  scale_color_brewer(palette="Dark2")+
  scale_fill_brewer(palette="Set2")+
  presettheme+
  facet_wrap(.~nestab_lab)

ggplot(we_plot_lambda,aes(x=theta,y=se_props,col=lambda_lab,fill=lambda_lab))+
  #geom_point(alpha=0.1)+
  geom_smooth(se=T)+
  labs(x=expression(""*theta*""),col=expression(""*lambda*""),fill=expression(""*lambda*""),y="Proportion Standard Error")+
  ggtitle(seprop_within)+
  scale_color_brewer(palette="Dark2")+
  scale_fill_brewer(palette="Set2")+
  presettheme+
  facet_wrap(.~nestab_lab)#,scale="free_y")


ggplot(we_plot_lambda,aes(x=theta,y=prop2digzeros,col=lambda_lab,fill=lambda_lab))+
  #geom_point(alpha=0.1)+
  geom_smooth(se=T)+
  labs(x=expression(""*theta*""),col=expression(""*lambda*""),fill=expression(""*lambda*""),y="Prop of Variables=0")+
  ggtitle(propzero_within)+
  scale_color_brewer(palette="Dark2")+
  scale_fill_brewer(palette="Set2")+
  presettheme+
  facet_wrap(.~nestab_lab)


nestabs_list=c(3,6,15,20,71,244)
niters=50
#niters=1
nreps=1

###scale dataframes
shapelist=c(1.5,4,10,30,50,70)#c(seq(1.5,9,0.5),seq(10,30,4),seq(40,70,10))
scalelist=c(seq(0.5,10,0.5),seq(10,30,2),seq(35,70,5))
combinations=base::expand.grid(nestabs_list,shapelist,scalelist)
#withinestab=data.frame()
acrossestab=data.frame(minval=numeric(),Q1val=numeric(),medianval=numeric(),
                       Q3val=numeric(),maxval=numeric(),spread=numeric(),
                       repid=integer(),iter=integer(),
                       shape=numeric(),scale=numeric(),nestab=integer(),nreps=integer())
#df=data.frame(prop=numeric(),estabid=integer(),iter=integer(),
#              shape=numeric(),scale=numeric(),nestab=integer(),nreps=integer())
set.seed(9)
for(i in seq(1,niters)){
  acrossestab_i=dplyr::bind_rows(apply(combinations,1,function(x)genprops(x[1],x[2],x[3],nreps=nreps,iter=i)))
  #withinestab_i=dplyr::bind_rows(lapply(temp,"[[",2))
  #acrossestab_i=dplyr::bind_rows(lapply(temp,"[[",3))
  #genprop_long_i=dplyr::bind_rows(lapply(temp,"[[",1))
  #withinestab=dplyr::bind_rows(list(withinestab,withinestab_i))
  acrossestab=dplyr::bind_rows(list(acrossestab,acrossestab_i))
  #df=dplyr::bind_rows(list(df,genprop_long_i))
}
acrossestab$nestab_lab=factor(paste0("Establishment Count=",acrossestab$nestab),levels=paste0("Establishment Count=",nestabs_list))
#withinestab$nestab_lab=factor(paste0("Establishment Count=",withinestab$nestab),levels=paste0("Establishment Count=",nestabs_list))
ae_scale=acrossestab
#we_scale=withinestab
ae_scale$shape_lab=paste0("shape=",ae_scale$shape)
ae_scale$shape_lab=factor(ae_scale$shape_lab,levels=paste0("shape=",shapelist))
#we_scale$shape_lab=paste0("shape=",we_scale$shape)
#we_scale$shape_lab=factor(we_scale$shape_lab,levels=paste0("shape=",shapelist))
str(ae_scale)

### shape data frames
shapelist=c(seq(1.5,9,0.5),seq(10,30,4),seq(40,70,10))
scalelist=c(0.5,1,2,4,10,30,50,70)#c(seq(0.5,10,0.5),seq(10,30,4),seq(40,70,10))
combinations=base::expand.grid(nestabs_list,shapelist,scalelist)
#withinestab=data.frame()
acrossestab=data.frame(minval=numeric(),Q1val=numeric(),medianval=numeric(),
                       Q3val=numeric(),maxval=numeric(),spread=numeric(),
                       repid=integer(),iter=integer(),
                       shape=numeric(),scale=numeric(),nestab=integer(),nreps=integer())
#df=data.frame(prop=numeric(),estabid=integer(),iter=integer(),
#              shape=numeric(),scale=numeric(),nestab=integer(),nreps=integer())
set.seed(9)
for(i in seq(1,niters)){
  acrossestab_i=dplyr::bind_rows(apply(combinations,1,function(x)genprops(x[1],x[2],x[3],nreps=nreps,iter=i)))
  #withinestab_i=dplyr::bind_rows(lapply(temp,"[[",2))
  #acrossestab_i=dplyr::bind_rows(lapply(temp,"[[",3))
  #genprop_long_i=dplyr::bind_rows(lapply(temp,"[[",1))
  #withinestab=dplyr::bind_rows(list(withinestab,withinestab_i))
  acrossestab=dplyr::bind_rows(list(acrossestab,acrossestab_i))
  #df=dplyr::bind_rows(list(df,genprop_long_i))
}
acrossestab$nestab_lab=factor(paste0("Establishment Count=",acrossestab$nestab),levels=paste0("Establishment Count=",nestabs_list))
#withinestab$nestab_lab=factor(paste0("Establishment Count=",withinestab$nestab),levels=paste0("Establishment Count=",nestabs_list))

ae_shape=acrossestab
#we_shape=withinestab
ae_shape$scale_lab=paste0("scale=",ae_shape$scale)
ae_shape$scale_lab=factor(ae_shape$scale_lab,levels=paste0("scale=",scalelist))
#we_shape$scale_lab=paste0("scale=",we_scale$shape)
#we_shape$scale_lab=factor(we_shape$scale_lab,levels=paste0("scale=",scalelist))

presettheme=theme_bw(base_size = 8)+theme(#axis.text.x = element_blank(),
  plot.title = element_text(size=7)#,
  #axis.title.x = element_blank(),
  )


ae_plot_scale=ae_scale[ae_scale$repid==1,]
# ggplot(ae_plot_scale,aes(x=scale,y=spread,color=shape_lab))+
#   #geom_point(alpha=0.25,size=0.2)+
#   geom_smooth(se=T)+
#   presettheme+
#   labs(x="Scale Parameter",y="Spread Across Establishments",color="Shape Parameter")+
#   ggtitle("Spread of Proportions across Establishments (within a variable) over Scale Parameters")+
#   scale_color_brewer(palette="Dark2")+
#   facet_wrap(~nestab_lab)

ae_plot_scale$se=ae_plot_scale$sd/ae_plot_scale$nestab
ggplot(ae_plot_scale,aes(x=scale,y=sd,color=shape_lab,fill=shape_lab))+
  #geom_point(alpha=0.25,size=0.2)+
  geom_smooth()+presettheme+
  labs(x="Scale Parameter",y="Standard Deviation Across Establishments",
       color="Shape Parameter",fill="Shape Parameter")+
  ggtitle("Standard Deviation across Establishments (within a variable) over Scale Parameters")+
  scale_color_brewer(palette="Dark2")+
  scale_fill_brewer(palette="Set2")+
  facet_wrap(~nestab_lab,scales="free_y")
ggplot(ae_plot_scale,aes(x=scale,y=prop2digzeros,color=shape_lab,fill=shape_lab))+
  #geom_point(alpha=0.25,size=0.2)+
  geom_smooth()+presettheme+
  labs(x="Scale Parameter",y="Proportion of Establishments=0",
       color="Shape Parameter",fill="Shape Parameter")+
  ggtitle("Proportion of Zeros across Establishments (within a variable) over Scale Parameters")+
  scale_color_brewer(palette="Dark2")+
  scale_fill_brewer(palette="Set2")+
  facet_wrap(~nestab_lab)

ae_plot_shape=ae_shape[ae_shape$repid==1,]
# ggplot(ae_plot_shape,aes(x=shape,y=spread,color=scale_lab))+
#   geom_point(alpha=0.25,size=0.2)+
#   geom_smooth(se=F)+presettheme+
#   labs(x="Shape Parameter",y="Spread Across Establishments",color="Scale Parameter")+
#   ggtitle("Spread of Proportions across Establishments (within a variable) over Scale Parameters")+
#   scale_color_brewer(palette="Set2")+
#   facet_wrap(~nestab_lab)

ggplot(ae_plot_shape,aes(x=shape,y=sd,color=scale_lab,fill=scale_lab))+
  #geom_point(alpha=0.25,size=0.2)+
  geom_smooth(se=T)+presettheme+
  labs(x="Shape Parameter",y="Standard Deviation Across Establishments",
       color="Scale Parameter",fill="Scale Parameter")+
  ggtitle("Standard Deviation across Establishments (within a variable) over Scale Parameters")+
  scale_color_brewer(palette="Dark2")+
  scale_fill_brewer(palette="Set2")+
  facet_wrap(~nestab_lab,scale="free_y")

ggplot(ae_plot_shape,aes(x=shape,y=prop2digzeros,color=scale_lab,fill=scale_lab))+
  #geom_point(alpha=0.25,size=0.2)+
  geom_smooth(se=T)+presettheme+
  labs(x="Shape Parameter",y="Proportion of Establishments=0",
       color="Scale Parameter",fill="Scale Parameter")+
  ggtitle("Proportion of Zeros across Establishments (within a variable) over Scale Parameters")+
  scale_color_brewer(palette="Dark2")+
  scale_fill_brewer(palette="Set2")+
  facet_wrap(~nestab_lab)

#########
nestabs_list=c(3,6,15,20,71,244)
nestab_list_lab_suffix=c("Q1","median","Q3","mean","95-percentile","99-percentile")
niters=50
#niters=1
nreps=4
###scale dataframes
shapelist=c(1.5,4,10,30,50,70)#c(seq(1.5,9,0.5),seq(10,30,4),seq(40,70,10))
scalelist=c(seq(0.5,10,0.5),seq(10,30,2),seq(35,70,5))
combinations=base::expand.grid(nestabs_list,shapelist,scalelist)
withinestab=data.frame()
# acrossestab=data.frame(minval=numeric(),Q1val=numeric(),medianval=numeric(),
#                        Q3val=numeric(),maxval=numeric(),spread=numeric(),
#                        repid=integer(),iter=integer(),
#                        shape=numeric(),scale=numeric(),nestab=integer(),nreps=integer())
#df=data.frame(prop=numeric(),estabid=integer(),iter=integer(),
#              shape=numeric(),scale=numeric(),nestab=integer(),nreps=integer())
set.seed(9)
for(i in seq(1,niters)){
  temp=dplyr::bind_rows(apply(combinations,1,function(x)genprops(x[1],x[2],x[3],nreps=nreps,iter=i)))
  withinestab_i=dplyr::bind_rows(lapply(temp,"[[",2))
  #acrossestab_i=dplyr::bind_rows(lapply(temp,"[[",3))
  #genprop_long_i=dplyr::bind_rows(lapply(temp,"[[",1))
  withinestab=dplyr::bind_rows(list(withinestab,withinestab_i))
  #acrossestab=dplyr::bind_rows(list(acrossestab,acrossestab_i))
  #df=dplyr::bind_rows(list(df,genprop_long_i))
}
#acrossestab$nestab_lab=factor(paste0("Establishment Count=",acrossestab$nestab),levels=paste0("Establishment Count=",nestabs_list))
col_nestab_lab_sfx=nestab_list_lab_suffix[sapply(withinestab$nestab,function(x)which(nestabs_list==x))]

withinestab$nestab_lab=paste0("Establishment Count=",withinestab$nestab," (",col_nestab_lab_sfx,")")
withinestab$nestab_lab=factor(paste0("Establishment Count=",withinestab$nestab),levels=paste0("Establishment Count=",nestabs_list," (",nestab_list_lab_suffix,")"))
#ae_scale=acrossestab
we_scale=withinestab
#ae_scale$shape_lab=paste0("shape=",ae_scale$shape)
#ae_scale$shape_lab=factor(ae_scale$shape_lab,levels=paste0("shape=",shapelist))
we_scale$shape_lab=paste0("shape=",we_scale$shape)
we_scale$shape_lab=factor(we_scale$shape_lab,levels=paste0("shape=",shapelist))
#str(ae_scale)

### shape data frames
shapelist=c(seq(1.5,9,0.5),seq(10,30,4),seq(40,70,10))
scalelist=c(0.5,1,2,4,10,20,30,50)#c(seq(0.5,10,0.5),seq(10,30,4),seq(40,70,10))
combinations=base::expand.grid(nestabs_list,shapelist,scalelist)
withinestab=data.frame()
# acrossestab=data.frame(minval=numeric(),Q1val=numeric(),medianval=numeric(),
#                        Q3val=numeric(),maxval=numeric(),spread=numeric(),
#                        repid=integer(),iter=integer(),
#                        shape=numeric(),scale=numeric(),nestab=integer(),nreps=integer())
#df=data.frame(prop=numeric(),estabid=integer(),iter=integer(),
#              shape=numeric(),scale=numeric(),nestab=integer(),nreps=integer())
set.seed(9)
for(i in seq(1,niters)){
  temp=dplyr::bind_rows(apply(combinations,1,function(x)genprops(x[1],x[2],x[3],nreps=nreps,iter=i)))
  withinestab_i=dplyr::bind_rows(lapply(temp,"[[",2))
  #acrossestab_i=dplyr::bind_rows(lapply(temp,"[[",3))
  #genprop_long_i=dplyr::bind_rows(lapply(temp,"[[",1))
  withinestab=dplyr::bind_rows(list(withinestab,withinestab_i))
  #acrossestab=dplyr::bind_rows(list(acrossestab,acrossestab_i))
  #df=dplyr::bind_rows(list(df,genprop_long_i))
}
#acrossestab$nestab_lab=factor(paste0("Establishment Count=",acrossestab$nestab),levels=paste0("Establishment Count=",nestabs_list))
col_nestab_lab_sfx=nestab_list_lab_suffix[sapply(withinestab$nestab,function(x)which(nestabs_list==x))]

withinestab$nestab_lab=paste0("Establishment Count=",withinestab$nestab," (",col_nestab_lab_sfx,")")
withinestab$nestab_lab=factor(paste0("Establishment Count=",withinestab$nestab),levels=paste0("Establishment Count=",nestabs_list," (",nestab_list_lab_suffix,")"))

#ae_shape=acrossestab
we_shape=withinestab
#ae_shape$scale_lab=paste0("scale=",ae_shape$scale)
#ae_shape$scale_lab=factor(ae_shape$scale_lab,levels=paste0("scale=",scalelist))
we_shape$scale_lab=paste0("scale=",we_scale$shape)
we_shape$scale_lab=factor(we_shape$scale_lab,levels=paste0("scale=",scalelist))

presettheme=theme_bw(base_size = 8)+theme(#axis.text.x = element_blank(),
  plot.title = element_text(size=7)#,
  #axis.title.x = element_blank(),
)


#ae_plot_scale=ae_scale[ae_scale$repid==1,]
we_plot_scale=we_scale[we_scale$estabid==1,]

# ggplot(ae_plot_scale,aes(x=scale,y=spread,color=shape_lab))+
#   #geom_point(alpha=0.25,size=0.2)+
#   geom_smooth(se=T)+
#   presettheme+
#   labs(x="Scale Parameter",y="Spread Across Establishments",color="Shape Parameter")+
#   ggtitle("Spread of Proportions across Establishments (within a variable) over Scale Parameters")+
#   scale_color_brewer(palette="Dark2")+
#   facet_wrap(~nestab_lab)

we_plot_scale$se=we_plot_scale$sd/we_plot_scale$nreps
ggplot(we_plot_scale,aes(x=scale,y=sd,color=shape_lab,fill=shape_lab))+
  #geom_point(alpha=0.25,size=0.2)+
  geom_smooth()+presettheme+
  labs(x="Scale Parameter",y="Standard Deviation Within Establishment",
       color="Shape Parameter",fill="Shape Parameter")+
  ggtitle("Standard Deviation within Establishment (across variables) over Scale Parameters")+
  scale_color_brewer(palette="Dark2")+
  scale_fill_brewer(palette="Set2")+
  facet_wrap(~nestab_lab,scales="free_y")
ggplot(we_plot_scale,aes(x=scale,y=prop2digzeros,color=shape_lab,fill=shape_lab))+
  #geom_point(alpha=0.25,size=0.2)+
  geom_smooth()+presettheme+
  labs(x="Scale Parameter",y="Proportion of Values=0",
       color="Shape Parameter",fill="Shape Parameter")+
  ggtitle("Proportion of Zeros within Establishment (across variables) over Scale Parameters")+
  scale_color_brewer(palette="Dark2")+
  scale_fill_brewer(palette="Set2")+
  facet_wrap(~nestab_lab)

we_plot_shape=we_shape[we_shape$estabid==1,]
# ggplot(ae_plot_shape,aes(x=shape,y=spread,color=scale_lab))+
#   geom_point(alpha=0.25,size=0.2)+
#   geom_smooth(se=F)+presettheme+
#   labs(x="Shape Parameter",y="Spread Across Establishments",color="Scale Parameter")+
#   ggtitle("Spread of Proportions across Establishments (within a variable) over Scale Parameters")+
#   scale_color_brewer(palette="Set2")+
#   facet_wrap(~nestab_lab)

ggplot(we_plot_shape,aes(x=shape,y=sd,color=scale_lab,fill=scale_lab))+
  #geom_point(alpha=0.25,size=0.2)+
  geom_smooth(se=T)+presettheme+
  labs(x="Shape Parameter",y="Standard Deviation Within Establishment",
       color="Scale Parameter",fill="Scale Parameter")+
  ggtitle("Standard Deviation a Establishments (within a variable) over Scale Parameters")+
  scale_color_brewer(palette="Dark2")+
  scale_fill_brewer(palette="Set2")+
  facet_wrap(~nestab_lab,scale="free_y")

ggplot(ae_plot_shape,aes(x=shape,y=prop2digzeros,color=scale_lab,fill=scale_lab))+
  #geom_point(alpha=0.25,size=0.2)+
  geom_smooth(se=T)+presettheme+
  labs(x="Shape Parameter",y="Proportion of Establishments=0",
       color="Scale Parameter",fill="Scale Parameter")+
  ggtitle("Proportion of Zeros across Establishments (within a variable) over Scale Parameters")+
  scale_color_brewer(palette="Dark2")+
  scale_fill_brewer(palette="Set2")+
  facet_wrap(~nestab_lab)


x=genprops(5,2,5)
x[[1]]
x[[2]]
x[[3]]

nreps=10000
nestab=10
presettheme=theme(axis.text.x = element_blank(),plot.title = element_text(size=7),axis.title.x = element_blank())


shapev=10
scalev=200
genprops=as.data.frame(MCMCprecision::rdirichlet(nreps,rgamma(nestab,shape=shapev,scale=scalev)))
genprops.long=tidyr::pivot_longer(genprops,cols=colnames(genprops),names_to = "estab",values_to = "prop")
genprops.long=dplyr::arrange(genprops.long,prop)
genprops.long$estabid=as.numeric(factor(genprops.long$estab,levels=unique(genprops.long$estab)))

df=genprops.long
df$shape=paste0("Shape=",shapev)
df$scale=paste0("Scale=",scalev)


shapev=1
genprops=as.data.frame(MCMCprecision::rdirichlet(nreps,rgamma(nestab,shape=shapev/nestab,scale=scalev)))
genprops.long=tidyr::pivot_longer(genprops,cols=colnames(genprops),names_to = "estab",values_to = "prop")
genprops.long=dplyr::arrange(genprops.long,prop)
genprops.long$estabid=as.numeric(factor(genprops.long$estab,levels=unique(genprops.long$estab)))
genprops.long$shape=paste0("Shape=",shapev)
genprops.long$scale=paste0("Scale=",scalev)
df=rbind(df,genprops.long)

shapev=20
genprops=as.data.frame(MCMCprecision::rdirichlet(nreps,rgamma(nestab,shape=shapev,scale=scalev)))
genprops.long=tidyr::pivot_longer(genprops,cols=colnames(genprops),names_to = "estab",values_to = "prop")
genprops.long=dplyr::arrange(genprops.long,prop)
genprops.long$estabid=as.numeric(factor(genprops.long$estab,levels=unique(genprops.long$estab)))
genprops.long$shape=paste0("Shape=",shapev)
genprops.long$scale=paste0("Scale=",scalev)

df=rbind(df,genprops.long)

shapev=10
scalev=50
genprops=as.data.frame(MCMCprecision::rdirichlet(nreps,rgamma(nestab,shape=shapev,scale=scalev)))
genprops.long=tidyr::pivot_longer(genprops,cols=colnames(genprops),names_to = "estab",values_to = "prop")
genprops.long=dplyr::arrange(genprops.long,prop)
genprops.long$estabid=as.numeric(factor(genprops.long$estab,levels=unique(genprops.long$estab)))
genprops.long$shape=paste0("Shape=",shapev)
genprops.long$scale=paste0("Scale=",scalev)

df=rbind(df,genprops.long)


shapev=1
genprops=as.data.frame(MCMCprecision::rdirichlet(nreps,rgamma(nestab,shape=shapev,scale=scalev)))
genprops.long=tidyr::pivot_longer(genprops,cols=colnames(genprops),names_to = "estab",values_to = "prop")
genprops.long=dplyr::arrange(genprops.long,prop)
genprops.long$estabid=as.numeric(factor(genprops.long$estab,levels=unique(genprops.long$estab)))
genprops.long$shape=paste0("Shape=",shapev)
genprops.long$scale=paste0("Scale=",scalev)

df=rbind(df,genprops.long)

shapev=20
genprops=as.data.frame(MCMCprecision::rdirichlet(nreps,rgamma(nestab,shape=shapev,scale=scalev)))
genprops.long=tidyr::pivot_longer(genprops,cols=colnames(genprops),names_to = "estab",values_to = "prop")
genprops.long=dplyr::arrange(genprops.long,prop)
genprops.long$estabid=as.numeric(factor(genprops.long$estab,levels=unique(genprops.long$estab)))
genprops.long$shape=paste0("Shape=",shapev)
genprops.long$scale=paste0("Scale=",scalev)

df=rbind(df,genprops.long)


shapev=10
scalev=350
genprops=as.data.frame(MCMCprecision::rdirichlet(nreps,rgamma(nestab,shape=shapev,scale=scalev)))
genprops.long=tidyr::pivot_longer(genprops,cols=colnames(genprops),names_to = "estab",values_to = "prop")
genprops.long=dplyr::arrange(genprops.long,prop)
genprops.long$estabid=as.numeric(factor(genprops.long$estab,levels=unique(genprops.long$estab)))

genprops.long$shape=paste0("Shape=",shapev)
genprops.long$scale=paste0("Scale=",scalev)

df=rbind(df,genprops.long)

plt1=ggplot(genprops.long,aes(y=prop,x=estabid,group=estabid))+
  geom_boxplot()+theme_minimal()+
  labs(y="generated proportion")+
  ggtitle(paste0("Shape=",shapev," Scale=",scalev))+presettheme


shapev=1
genprops=as.data.frame(MCMCprecision::rdirichlet(nreps,rgamma(nestab,shape=shapev,scale=scalev)))
genprops.long=tidyr::pivot_longer(genprops,cols=colnames(genprops),names_to = "estab",values_to = "prop")
genprops.long=dplyr::arrange(genprops.long,prop)
genprops.long$estabid=as.numeric(factor(genprops.long$estab,levels=unique(genprops.long$estab)))
genprops.long$shape=paste0("Shape=",shapev)
genprops.long$scale=paste0("Scale=",scalev)

df=rbind(df,genprops.long)
plt2=ggplot(genprops.long,aes(y=prop,x=estabid,group=estabid))+
  geom_boxplot()+theme_minimal()+
  labs(y="generated proportion")+
  ggtitle(paste0("Shape=",shapev," Scale=",scalev))+presettheme

shapev=20
genprops=as.data.frame(MCMCprecision::rdirichlet(nreps,rgamma(nestab,shape=shapev,scale=scalev)))
genprops.long=tidyr::pivot_longer(genprops,cols=colnames(genprops),names_to = "estab",values_to = "prop")
genprops.long=dplyr::arrange(genprops.long,prop)
genprops.long$estabid=as.numeric(factor(genprops.long$estab,levels=unique(genprops.long$estab)))
genprops.long$shape=paste0("Shape=",shapev)
genprops.long$scale=paste0("Scale=",scalev)

df=rbind(df,genprops.long)

df$shape=factor(as.character(df$shape),levels=paste0("Shape=",c(1,10,20)))
df$scale=factor(as.character(df$scale),levels=paste0("Scale=",c(50,200,350)))

ggplot(df,aes(y=prop,x=estabid,group=estabid))+geom_boxplot()+theme_minimal()+
  labs(y="proportion")+presettheme+facet_grid(shape~scale)


nreps=6
nestab=5
shapev=10
scalev=200
genprops=as.data.frame(MCMCprecision::rdirichlet(nreps,rgamma(nestab,shape=shapev,scale=scalev)))
head(genprops)



##### Check Calculations... ####
nval=100
lambdaval=2
thetaval=5
sampsval=3000

approxEfunctgamma=function(lambda,theta,n){
  firstterm=1/(1+(lambda*theta))
  secondterm=(lambda*theta)/(((1+(lambda*theta)))^3)
  firstterm+secondterm
}

theory_var_approx=function(lambda=lambdaval,theta=thetaval,n=nval){
  ncoef=(n-1)/((n^2)*(lambda+1))
  multterm=1+(lambda*approxEfunctgamma(lambda=lambda,theta=theta,n=n))
  ncoef*multterm
}

theory_covar_approx=function(lambda=lambdaval,theta=thetaval,n=nval){
  ncoef=(-1)/((n^2)*(lambda+1))
  multterm=1+(lambda*approxEfunctgamma(lambda=lambda,theta=theta,n=n))
  ncoef*multterm
}

gen_dir_gamma_prior=function(lambda=lambdaval,theta=thetaval,n=nval,samps=sampsval,only_dif_theory=F){
  gam_prior=rgamma(n=n,shape=lambda/n,scale=theta)
  rdir=MCMCpack::rdirichlet(samps,gam_prior)
  Vmat=var(rdir)
  samp_vars=sapply(seq(1,n),function(i)var(rdir[,i])) #var across samples
  samp_covars=cov(rdir) #covariance within a sample
  samp_within_vars=sapply(seq(1,samps),function(i)var(rdir[i,]))
  if(only_dif_theory==T){
    return(list("vars_mean_dif"=(samp_vars-theory_var_approx(lambda=lambda,theta=theta,n=n))[seq(2,n-1)]/(n-1),
                "covars_mean_dif"=samp_covars[1,seq(2,n-1)]-theory_covar_approx(lambda=lambda,theta=theta,n=n)))
  }else{
    return(list("samples"=rdir,"samp_vars"=samp_vars,"samp_covars"=samp_covars[1,seq(1,n)],"samp_win_var"=samp_within_vars))
  }
}
temp=gen_dir_gamma_prior(1.50001,5,5,100)

df=data.frame("variance_establishment_across_samples"=temp[2],
              "covariance"=temp[3],
              "variance_establishments_within_sample"=temp[4])
summary(df)

temp[2]           
temp[3]
temp[4]



fulldata=data.frame()
samps=4
nval=6
set.seed(5)
for(lambda in c(1.0001,2.0001,5.0001)){
  for(theta in c(0.5,1,2,5)){
    temp=gen_dir_gamma_prior(lambda=lambda,theta=theta,n=nval,samps=samps,only_dif_theory = F)
    dfmatrix=temp[1]
    rwvalues=as.data.frame(dfmatrix)
    colnames(rwvalues)=paste0("estab",as.character(seq(1,nval)))
    rwvalues$iter=seq(1,samps)
    rwvalues=tidyr::pivot_longer(rwvalues,cols=paste0("estab",as.character(seq(1,nval))),names_to="estab",values_to = "value")
    rwvalues$shape=paste0("lambda=",as.character(lambda))
    rwvalues$scale=paste0("theta=",as.character(theta))
    fulldata=rbind(fulldata,rwvalues)
  }
}

head(fulldata)
ggplot2::ggplot(fulldata,ggplot2::aes(y=value,x=estab,group=estab,color=estab))+
  ggplot2::geom_boxplot()+
  ggplot2::labs(x="Establishment in County by NAICS-6 Cell",y="Proportion Allocated",
                title="Dirichlet Divider with Gamma Prior (number of establishments=6)")+
  ggplot2::scale_color_brewer(palette="Dark2")+
  ggplot2::theme_bw(base_size=8)+
  ggplot2::theme(legend.position="none")+
  ggplot2::facet_grid(scale~shape)
