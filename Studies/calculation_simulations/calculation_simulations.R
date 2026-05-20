require("MCMCpack")
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

gen_dir_gamma_prior=function(lambda=lambdaval,theta=thetaval,n=nval,samps=sampsval,only_dif_theory=T){
  gam_prior=rgamma(n=n,shape=lambda/n,scale=theta)
  rdir=MCMCpack::rdirichlet(samps,gam_prior)
  samp_vars=sapply(seq(1,n),function(i)var(rdir[,i])) #var across samples
  samp_covars=cov(rdir) #covariance within a sample
  if(only_dif_theory==T){
    return(list("vars_mean_dif"=(samp_vars-theory_var_approx(lambda=lambda,theta=theta,n=n))[seq(2,n-1)]/(n-1),
                "covars_mean_dif"=samp_covars[1,seq(2,n-1)]-theory_covar_approx(lambda=lambda,theta=theta,n=n)))
  }else{
    return(list("samples"=rdir,"samp_vars"=samp_vars,"samp_covars"=samp_covars[1,seq(2,n)]))
  }
}

for(lambda in c(1.0001,2.0001,5.0001,10.001)){
  for(theta in c(0.5,1,2,5,10)){
    temp=gen_dir_gamma_prior(lambda=lambda,theta=theta,n=nval,samps=sampsval,only_dif_theory = T)
    dftemp=data.frame("vars_difs"=temp[[1]],"covars_difs"=temp[[2]])
    print(paste0("lambda=",lambda,", theta=",theta))
    print(summary(dftemp))
  }
}
temp=gen_dir_gamma_prior()
dftemp=data.frame("vars_difs"=temp[[1]],"covars_difs"=temp[[2]])
summary(dftemp)
theory_var_approx()
summary(temp[[]])
theory_covar_approx()

