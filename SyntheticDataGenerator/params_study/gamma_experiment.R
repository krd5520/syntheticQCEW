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
library(parameters)


############# Functions ############
across_stats=function(x,nestab,name_sfx="_props"){
  vec=c(quantile(x,na.rm=T,probs=c(0,0.5,1)),
        sd(x,na.rm=T),sqrt(var(x,na.rm=T)/nestab),
        parameters::skewness(x,na.rm=T,type="2")$Skewness,
        parameters::kurtosis(x,na.rm=T,type="2")$Kurtosis)
  vec
}
within_stats=function(x,nreps,name_sfx="_props"){
  vec=c(quantile(x,na.rm=T,probs=c(0,0.5,1)),
        sd(x,na.rm=T),sqrt(var(x,na.rm=T)/nreps),
        parameters::skewness(x,na.rm=T,type="2")$Skewness,
        parameters::kurtosis(x,na.rm=T,type="2")$Kurtosis,
        sd(x[1:3],na.rm=T),
        parameters::skewness(x[1:3],na.rm=T,type="2")$Skewness,
        parameters::kurtosis(x[1:3],na.rm=T,type="2")$Kurtosis)
  
  vec
}

##################################################################
### Generate Across and Within Establishment Summaries ##########
#################################################################

genprops=function(nestab,shapev,scalev,nreps=4,iter=1,
                  avg_per_estab=c(15,15,15,165000),
                  lambda=NULL,theta=NULL,
                  lambda_funct=NULL,theta_funct=NULL,wagemin=1,
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
  genpropsdf=genpropsdf[seq(1,nreps),]
  gencountdf=gencountdf[seq(1,nreps),]
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

  genprops_add_stats=function(df,zeros=NULL,numreps=nreps,shapeval=shapev,scaleval=scalev,
                              i_iteration=iter,numestab=nestab,lambdaval=lambda,thetaval=theta,iterid=all_iter_id,
                              lambdaf=lambda_funct,thetaf=theta_funct){
    df$spread_props=df$maxval_props-df$minval_props
    df$spread_counts=df$maxval_counts-df$minval_counts
    df$count2digzeros=zeros
    df$prop2digzeros=zeros/numestab
    if("lambda" %in% colnames(df)){
      return(df)
    }else{
      df$nestab=numestab
      df$nreps=numreps
      df$iter=i_iteration
      df$lambda=lambdaval
      df$theta=thetaval
      df$lambda_funct=lambdaf
      df$theta_funct=thetaf
      df$iterid=iterid
      return(df)
    }
  }
  ac_rzeros=rowSums(round(gencountdf,0)==0)
  
  acrossestab_summary_props=as.data.frame(apply(genpropsdf,1,function(x)across_stats(x=x,nestab=nestab,name_sfx = "_preprops")),
                                          row.names=paste0(c("minval","medianval","maxval","sd","se","skew","kurtosis"),"_preprops")#name_sfx)paste0(c("minval","Q1val","medianval","Q3val","maxval","mean","sd","se"),"_props")
)
  acrossestab_summary_counts=as.data.frame(apply(gencountdf,1,function(x)across_stats(x=x,nestab=nestab,name_sfx = "_counts")),
                                           row.names = paste0(c("minval","medianval","maxval","sd","se","skew","kurtosis"),"_counts")#paste0(c("minval","Q1val","medianval","Q3val","maxval","mean","sd","se"),"_counts")
  )
  colnames(acrossestab_summary_counts)=paste0("rep",seq(1,nreps))
  colnames(acrossestab_summary_props)=paste0("rep",seq(1,nreps))
  
  
    gencountprop=gencountdf/(avg_per_estab*nestab)
    postwagemin=as.data.frame(apply(gencountprop,1,function(x)across_stats(x=x,nestab=nestab,name_sfx="_props")),
                              row.names = paste0(c("minval","medianval","maxval","sd","se","skew","kurtosis"),"_props"))
    colnames(postwagemin)=paste0("rep",seq(1,nreps))
    dflist=list(acrossestab_summary_props,acrossestab_summary_counts,postwagemin)
    for(i in seq(1,3)){
      df=dflist[[i]]
      df$rn=rownames(df)
      dflist[[i]]=df
    }
    
    acrossestab_summary=data.table::transpose(data.table::rbindlist(dflist,fill=T), make.names = "rn")
 
  acrossestab_summary=genprops_add_stats(df=acrossestab_summary,zeros=ac_rzeros)
  acrossestab_summary$repid=seq(1,nreps)
  across_reduced_cols= c(paste0(c("maxval","sd","se","skew","kurtosis"),"_props"),
                         paste0(c("sd","se","skew","kurtosis"),"_counts"),
                         "repid","lambda","theta","nestab","spread_counts",
                         "spread_props","prop2digzeros","iterid")
  acrossestab_summary$scope="across"

  if(nreps>1){
    wi_czeros=colSums(round(gencountdf,0)==0)
    withinestab_summary_props=t(apply(genpropsdf,2,function(x)within_stats(x,nreps=nreps,name_sfx="_preprops")))
    colnames(withinestab_summary_props)=paste0(c("minval","medianval","maxval",
                                                 "sd","se","skew","kurtosis",
                                                 "sd_emp","skew_emp","kurtosis_emp"),"_preprops")
    withinestab_summary_counts=t(apply(gencountdf,2,function(x)within_stats(x,nreps=nreps,name_sfx="_counts")))
    colnames(withinestab_summary_counts)=paste0(c("minval","medianval","maxval",
                                                 "sd","se","skew","kurtosis",
                                                 "sd_emp","skew_emp","kurtosis_emp"),"_counts")
    
    postwagemin=t(apply(gencountprop,2,function(x)within_stats(x=x,nreps=nreps,name_sfx="_props")))
    colnames(postwagemin)=paste0(c("minval","medianval","maxval","sd","se","skew","kurtosis",
                                   "sd_emp","skew_emp","kurtosis_emp"),"_props")
    withinestab_summary=dplyr::bind_cols(list(as.data.frame(withinestab_summary_counts),
                                              as.data.frame(withinestab_summary_props),
                                              as.data.frame(postwagemin)))
    rownames(withinestab_summary)=paste0("estab",seq(1,nestab))
    withinestab_summary=as.data.frame(withinestab_summary)
    withinestab_summary$estabid=seq(1,nestab)
    withinestab_summary=genprops_add_stats(df=withinestab_summary,zeros=wi_czeros)
    withinestab_summary$prop2digzeros=wi_czeros/nreps
    withinestab_summary$count2digzeros=wi_czeros
    
    
    if(return_reduced==T){
      withinestab_summary$scope="within"
      within_reduced_cols= c(paste0(c("maxval","sd","se","sd_emp","skew","kurtosis","skew_emp","kurtosis_emp"),"_props"),
                             paste0(c("sd","se","sd_emp","skew","kurtosis","skew_emp","kurtosis_emp"),"_counts"),
                             "estabid","lambda","theta","nestab",
                             "spread_props","prop2digzeros","count2digzeros","iterid")
      reduced_across=base::subset(acrossestab_summary,
                                  subset=((acrossestab_summary$repid==1)|(acrossestab_summary$repid==4)),
                                  select=across_reduced_cols)
      reduced_within=base::subset(withinestab_summary,
                                  subset=(withinestab_summary$estabid==1),
                                  select=within_reduced_cols)

     fulldf=data.table::rbindlist(
          list(reduced_across,reduced_within),fill=T)
     #print(head(fulldf))
     return(fulldf)
      
    }else{
      return(list(acrossestab_summary,withinestab_summary))
    }
  }else{
    if(return_reduced==T){ 
      return(acrossestab_summary[,colnames(acrossestab_summary)%in%across_reduced_cols])
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

 


### Plotting functions

#General plotting functions
 

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
  gp$grobs[[guide.idx]]=zeroGrob()
  return(gp)
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

 


 
plot_titler=function(xvar,yvarlab,colvar,across=T,shape_transform="identity",
                     scale_transform="identity",
                     plottype="smooth",
                     add_avg_transform=NULL,
                     title_stem=NULL){
  acrosstitle=paste0(ifelse(across==T,"Across","Within")," Establishments")
  titleparts=c(", for Prior Gamma(Shape=",", Scale=")
  titlestart=bquote(.(yvarlab)~.(acrosstitle))
  shape_part=bquote("Shape"==lambda)
  scale_part=bquote(", Scale"==theta)
  
  if(scale_transform=="over_lambda"){
    scale_part=bquote(", Scale"==frac(theta,lambda))
  }else if(scale_transform=="over_lambda_over_d"){
    scale_part=bquote(", Scale"==d*frac(theta,lambda))
  }
  if(shape_transform=="over_d"){
    shape_part=bquote("Shape"==frac(lambda,d))
  }
  prior_start=bquote("for Priors Gamma("*.(shape_part)*.(scale_part)*")")
  if(is.null(add_avg_transform)==F){
    if((is.numeric(add_avg_transform)==T)&(is.na(add_avg_transform)==F)){
      priortitle=bquote(.(prior_start)+("Avg"==.(add_avg_transform)))
    }else if((is.character(add_avg_transform)==T)&(add_avg_transform!="")){
      priortitle=bquote(.(prior_start)+("Avg"==.(add_avg_transform)))
    }else if((is.logical(add_avg_transform)==T)&(add_avg_transform==T)){
      priortitle=bquote(.(prior_start)+("Avg"))
}
    }else{
    priortitle=prior_start
  }
  if(is.null(title_stem)==F){
    titleexpr=bquote(.(titlestart)~.(prior_start)~.(title_stem))
  }else{
    titleexpr=bquote(.(titlestart)~.(prior_start))
  }
  
  if (is.null(titleexpr)){titleexpr <- paste(ifelse(across==T,"Across Establishment ","Within Establishment "), yvar)}
  
  return(titleexpr)
  
}

cb_color_fill_picker=function(nlev){
  if(nlev<9){
    linecols=RColorBrewer::brewer.pal(max(nlev,3),"Dark2")[seq_len(nlev)]
    fillcols=RColorBrewer::brewer.pal(max(nlev,3),"Set2")[seq_len(nlev)]
  }else{
    linecols=viridis::viridis(nlev)#c(RColorBrewer::brewer.pal(8,"Dark2"),RColorBrewer::brewer.pal(max(3,nlev-8),"Set2")[seq_len(nlev-8)])
    fillcols=colorspace::lighten(linecols,amount=0.3)
  }
  return(list(linecols,fillcols))
}

y_labeler=function(yvar,across=T){
  ylab_lookup <- c(
    se_counts      = "Count Standard Error",
    se_props       = "Proportion Standard Error",
    spread_props=" Range of Proportions",
    sd_props="Proportion Standard Deviation",
    sd_counts="Count Standard Deviation",
    sd_emp_counts="Employment Count Standard Deviation",
    skew_counts="Count Skewness",
    skew_props="Proportion Skewness",
    kurtosis_counts="Count Kurtosis",
    kurtosis_props="Proportion Kurtosis"
  )
  if(across==T){
    ylab_lookup=c(ylab_lookup,c(prop2digzeros  = "Proportion of Establishment=0",
                                count2digzeros = "Count of Establishment=0"))
  }else{
    ylab_lookup=c(ylab_lookup,c(prop2digzeros  = "Proportion of Values=0",
                                count2digzeros = "Count of Values=0"))
  }
  yvarlabel=ylab_lookup[yvar]
  if (is.na(yvarlabel)){yvarlabel = yvar}   # fallback
  return(yvarlabel)
}
 



 
plot_gamma_smooth <- function(data,xvar,yvar,colvar,facetvar   = "nestab_lab",
                              across=T,
                              free_y      = FALSE,
                              nrow_legend = 3,
                              savefolder   = NULL,fname_stem=NULL,
                              width       = 7.7,height      = 3,units       = "in",
                              shape_transform="identity",scale_transform="identity",add_avg_transform=NULL,
                              include_points=F,jitterwidth=1.5,
                              title_stem=NULL,
                              addtheme=NULL) {
  ### 
  # Label for y variable
  yvarlab=y_labeler(yvar,across=across)
  titleexpr=plot_titler(xvar=xvar,yvarlab=yvarlab,colvar=colvar,across=across,
                        shape_transform=shape_transform,scale_transform=scale_transform,
                        add_avg_transform=add_avg_transform,title_stem=title_stem)
  
  lev =levels(data[[colvar]])
  data[[colvar]]=factor(as.character(data[[colvar]]),levels=lev[lev %in% unique(as.character(data[[colvar]]))])
  lev =levels(data[[colvar]])
  if (is.null(lev)){lev=sort(unique(data[[colvar]]))}   # character fallback
  parsedlabs = parse(text = lev)   # theta==1 renders correctly as θ=1
  
  nlev=length(lev)
  tempcols=cb_color_fill_picker(nlev)
  linecols=tempcols[[1]]
  fillcols=tempcols[[2]]
  
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
                    position=position_jitter(width=jitterwidth))
    #position_jitterdodge(dodge.width=1,#0.5,jitter.width=1.5))#0.1))
  }
  p<-p+
    geom_smooth(se = TRUE,method = "loess",formula = 'y ~ x',alpha=0.3) +
    labs(x = xexpr, y = yvarlab) +
    ggtitle(titleexpr)+
    scale_color_manual(
      name    = colexpr,
      values = linecols,
      labels  = parsedlabs
    ) +
    scale_fill_manual(
      name    = colexpr,        # must match scale_color_brewer name
      values = fillcols,
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

#########################################################################################


#Constant Parameters
nestabs_list=c(3,6,15,20,71)#,244)
nestab_labs_list=c("Q1","median","Q3","mean","95-percentile")#,"99-percentile")
avg_perestnum=c(15,15,15,165000) #avg emp1 and avg wages rounded
nreps=4
presettheme=ggplot2::theme_bw(base_size = 8)+ggplot2::theme(#axis.text.x = element_blank(),
  plot.title = ggplot2::element_text(size=7)#,
  #axis.title.x = element_blank(),
)

## Graphic settings
ae_free_y_lookup=c(se_counts      = FALSE,se_props       = FALSE,
  prop2digzeros  = FALSE,count2digzeros = FALSE,
  spread_props=FALSE,
  sd_props=FALSE,sd_counts=FALSE,
  kurtosis_props=FALSE,kurtosis_counts=FALSE,
  skew_props=FALSE,skew_counts=FALSE,)
we_free_y_lookup=c(se_counts      = FALSE,se_props       = TRUE,
  prop2digzeros  = FALSE,count2digzeros = FALSE,
  spread_props=TRUE,
  sd_props=TRUE,sd_emp_counts=FALSE)

#######################################################
########### Original/Basic Generation
############################################################
niters=100
lambda_list=c(1.5,2.5,5,7.5,10,15,20,25,30,40,50,60,70,80,90,100)
theta_list=c(0.25,0.5,0.75,1,2,4,6,8,10,15,20,25,30,40,50,75,100)
#niters=5
#lambda_list=c(1.5,1,2.5,5,7.5,10,15)
#theta_list=c(1,2,4,6,8,10,20,30,40,50)
combination=base::expand.grid(nestabs_list,lambda_list,theta_list,seq(1,niters))

combination[,5]=combination[,2]/combination[,1]
combination[,6]=combination[,3]
colnames(combination)=c("nestab","lambda","theta","iter","shape","scale")

set.seed(9)
mclapplytemp=parallel::mclapply(seq(1,nrow(combination)),
                        function(rw)genprops(combination[rw,1],combination[rw,5],combination[rw,6],
                                             nreps=nreps,iter=combination[rw,4],avg_per_estab=avg_perestnum,
                                             lambda=combination[rw,2],theta=combination[rw,3],
                                             lambda_funct="over_d",theta_funct="identity"))
fulldf_og=data.table::rbindlist(mclapplytemp,fill=T)
fulldf_og=add_labels_data(fulldf_og,
                          nestabs_list,nestab_labs_list,
                          theta_list,lambda_list)
withinestab_og=fulldf_og[fulldf_og$scope=="within",]
acrossestab_og=fulldf_og[fulldf_og$scope=="across",]

og_fstem="shape_lambda_over_d_scale_theta_wagemin_iter100"
dir.create(og_fstem)
saveRDS(withinestab_og,file=paste0(og_fstem,"/",og_fstem,"_withinestab_og.rds"))
saveRDS(acrossestab_og,file=paste0(og_fstem,"/",og_fstem,"_acrossestab_og.rds"))
#############

theta_discrete= c(0.25,0.5,1,2,5,10,20,50,100)
theta_range=c(0.75,51)
lambda_discrete=c(1.5,2.5,5,10,20,30,50,75,100)
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


for(vstr in c("prop2digzeros","spread_props","sd_counts","sd_props","skew_counts","kurtosis_counts")){
  if(vstr!="prop2digzero"){
  plot_gamma_smooth(ae_plot_theta_og[ae_plot_theta_og$repid==4,],"lambda",vstr,
                    colvar="theta_lab",across=T,free_y=ae_free_y_lookup[vstr],
                    savefolder = og_fstem,fname_stem=paste0(og_fstem,"_wages"),
                    shape_transform = "over_d",scale_transform="identity",add_avg_transform = NULL,include_points=F,
                    title_stem = "(Quarterly Wages, min(wages)=1)")#ifelse(vstr%in%c("prop2digzeros","spread_props"),T,F))
    plot_gamma_smooth(ae_plot_lambda_og[ae_plot_lambda_og$repid==4,],"theta",vstr,
                      colvar="lambda_lab",across=T,free_y=ae_free_y_lookup[vstr],
                      savefolder = og_fstem,fname_stem=paste0(og_fstem,"_wages"),
                      shape_transform = "over_d",scale_transform="identity",add_avg_transform = NULL,
                      include_points=F,title_stem = "(Quarterly Wages, min(wages)=1)")
    
  }
  plot_gamma_smooth(ae_plot_theta_og[ae_plot_theta_og$repid==1,],"lambda",vstr,
                    colvar="theta_lab",across=T,free_y=ae_free_y_lookup[vstr],
                    savefolder = og_fstem,fname_stem=paste0(og_fstem,"_emp1"),
                    shape_transform = "over_d",scale_transform="identity",add_avg_transform = NULL,
                    include_points=F,title_stem="(Month 1 Employment)")#ifelse(vstr%in%c("prop2digzeros","spread_props"),T,F))
  plot_gamma_smooth(ae_plot_lambda_og[ae_plot_lambda_og$repid==1,],"theta",vstr,
                    colvar="lambda_lab",across=T,free_y=ae_free_y_lookup[vstr],
                    savefolder = og_fstem,fname_stem=paste0(og_fstem,"_emp1"),
                    shape_transform = "over_d",scale_transform="identity",add_avg_transform = NULL,
                    include_points=F,title_stem = "(Month 1 Employement)")#ifelse(vstr%in%c("prop2digzeros","spread_props"),T,F))
 }

for(vstr in c("sd_props","spread_props","count2digzeros","sd_emp_counts","skew_props","kurtosis_props")){
  plot_gamma_smooth(we_plot_theta_og,"lambda",vstr,
                    colvar="theta_lab",across=F,free_y=we_free_y_lookup[vstr],
                    savefolder = og_fstem,fname_stem=og_fstem,
                    shape_transform = "over_d",scale_transform="identity",add_avg_transform = NULL,
                    include_points=F)#ifelse(vstr%in%c("prop2digzeros","count2digzeros","spread_props"),T,F))
  plot_gamma_smooth(we_plot_lambda_og,"theta",vstr,
                    colvar="lambda_lab",across=F,free_y=we_free_y_lookup[vstr],
                    savefolder = og_fstem,fname_stem=og_fstem,
                    shape_transform = "over_d",scale_transform="identity",add_avg_transform = NULL,
                    include_points=F)#ifelse(vstr%in%c("prop2digzeros","count2digzeros","spread_props"),T,F))
}


####### Experiment #######
#niters=100
#lambda_list=c(1.5,2.5,5,10,15,20,30,40,50,75)#c(1.5,2.5,5,7.5,10,15,20,30,40,50,60,70,85,100)
#theta_list=c(0.5,1,3,5,10,15,20,25,30,40,50)#c(0.1,0.5,1,2,4,6,8,10,15,20,25,30,40,50)
combination_exp=base::expand.grid(nestabs_list,lambda_list,theta_list,seq(1,niters))

combination_exp[,5]=combination_exp[,2]/combination_exp[,1]
combination_exp[,6]=combination_exp[,3]
colnames(combination_exp)=c("nestab","lambda","theta","iter","shape","scale")



set.seed(9)

exp_df_list=list()
temp=parallel::mclapply(seq(1,nrow(combination_exp)),
                        function(rw)genprops(combination_exp[rw,1],combination_exp[rw,5],combination_exp[rw,6],
                                             nreps=nreps,iter=combination_exp[rw,4],avg_per_estab=avg_perestnum,
                                             lambda=combination_exp[rw,2],theta=combination_exp[rw,3],
                                             lambda_funct="over_d",theta_funct="identity",experiment = T))
fulldf_exp=data.table::rbindlist(temp,fill=T)
withinestab_exp=fulldf_exp[fulldf_exp$scope=="within",]
acrossestab_exp=fulldf_exp[fulldf_exp$scope=="across",]
withinestab_exp=add_labels_data(withinestab_exp,nestabs_list,nestab_labs_list,theta_list,lambda_list)
acrossestab_exp=add_labels_data(acrossestab_exp,nestabs_list,nestab_labs_list,theta_list,lambda_list)
#############

#theta_list=c(0.1,0.5,1,5,10,15,20,30,40,50,75,100)
ae_plot_theta_exp=acrossestab_exp[(acrossestab_exp$theta %in% theta_discrete)&
                                  (acrossestab_exp$lambda<lambda_range[2])&
                                  (acrossestab_exp$lambda>lambda_range[1]),]
we_plot_theta_exp=withinestab_exp[(withinestab_exp$theta %in% theta_discrete)&
                                  (withinestab_exp$lambda<lambda_range[2])&
                                  (withinestab_exp$lambda>lambda_range[1]),]
#lambda_list=c(1.5,5,10,15,20,30,40,50,75,100)
ae_plot_lambda_exp=acrossestab_exp[(acrossestab_exp$lambda %in% lambda_discrete)&
                                   (acrossestab_exp$theta<theta_range[2])&
                                   (acrossestab_exp$theta>theta_range[1]),]
we_plot_lambda_exp=withinestab_exp[(withinestab_exp$lambda %in% lambda_discrete)&
                                   (withinestab_exp$theta<theta_range[2])&
                                   (withinestab_exp$theta>theta_range[1]),]

exp_fstem="shape_lambda_over_d_scale_theta_wagemin_iter100_experimental"
dir.create(exp_fstem)
saveRDS(withinestab_exp,file=paste0(exp_fstem,"/",exp_fstem,"_withinestab_exp.rds"))
saveRDS(acrossestab_exp,file=paste0(exp_fstem,"/",exp_fstem,"_acrossestab_exp.rds"))

for(vstr in c("prop2digzeros","spread_props","sd_counts","sd_props")){
  plot_gamma_smooth(ae_plot_theta_exp[ae_plot_theta_exp$repid==4,],"lambda",vstr,
                    colvar="theta_lab",across=T,free_y=ae_free_y_lookup[vstr],
                    savefolder = exp_fstem,
                    fname_stem=paste0(exp_fstem,"_wages"),
                    shape_transform = "over_d",
                    scale_transform="identity",
                    add_avg_transform=avg_perestnum[4],
                    include_points=F,
                    title_stem = "(Quarterly Wages)")#ifelse(vstr%in%c("prop2digzeros","spread_props"),T,F))
  plot_gamma_smooth(ae_plot_theta_exp[ae_plot_theta_exp$repid==1,],"lambda",vstr,
                    colvar="theta_lab",across=T,free_y=ae_free_y_lookup[vstr],
                    savefolder = exp_fstem,
                    fname_stem=paste0(exp_fstem,"_emp1"),
                    shape_transform = "over_d",
                    scale_transform="identity",
                    add_avg_transform=avg_perestnum[1],
                    include_points=F,
                    title_stem="(Month 1 Employment)")#ifelse(vstr%in%c("prop2digzeros","spread_props"),T,F))
  plot_gamma_smooth(ae_plot_lambda_exp[ae_plot_lambda_exp$repid==1,],"theta",vstr,
                    colvar="lambda_lab",across=T,free_y=ae_free_y_lookup[vstr],
                    savefolder = exp_fstem,
                    fname_stem=paste0(exp_fstem,"_emp1"),
                    shape_transform = "over_d",
                    scale_transform="identity",
                    add_avg_transform=avg_perestnum[1],
                    include_points=F,title_stem = "(Month 1 Employement)")#ifelse(vstr%in%c("prop2digzeros","spread_props"),T,F))
  plot_gamma_smooth(ae_plot_lambda_exp[ae_plot_lambda_exp$repid==4,],"theta",vstr,
                    colvar="lambda_lab",across=T,free_y=ae_free_y_lookup[vstr],
                    savefolder = exp_fstem,
                    fname_stem=paste0(exp_fstem,"_wages"),
                    shape_transform = "over_d",
                    scale_transform="identity",
                    add_avg_transform=avg_perestnum[4],
                    include_points=F,title_stem = "(Quarterly Wages)")
}

for(vstr in c("sd_props","spread_props","count2digzeros","sd_emp_counts")){
  plot_gamma_smooth(we_plot_theta_exp,"lambda",vstr,
                    colvar="theta_lab",across=F,free_y=we_free_y_lookup[vstr],
                    savefolder = exp_fstem,
                    fname_stem=exp_fstem,
                    shape_transform = "over_d",
                    scale_transform="identity",
                    add_avg_transform=avg_perestnum[4],
                    include_points=F)#ifelse(vstr%in%c("prop2digzeros","count2digzeros","spread_props"),T,F))
  plot_gamma_smooth(we_plot_lambda_exp,"theta",vstr,
                    colvar="lambda_lab",across=F,free_y=we_free_y_lookup[vstr],
                    savefolder = exp_fstem,
                    fname_stem=exp_fstem,
                    shape_transform = "over_d",
                    scale_transform="identity",
                    include_points=F)#ifelse(vstr%in%c("prop2digzeros","count2digzeros","spread_props"),T,F))
}

