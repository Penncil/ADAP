####################
# analysis functions 
####################
#compute P(y_i=1|x_i)
expit <- function(x){exp(x)/(1+exp(x))}

#soft-thresholding operator
soft <- function(x,thres){
  if (x > thres){
    res <- x - thres
  } else if (x < -thres){
    res <- x + thres
  } else {
    res <- 0
  }
  res
}

#coordinate descent algorithm
coordi <- function(atilde, B, betainit, lambda){
  tol <- 1e-4
  loop <- 1
  Loop <- 100
  msg <- NA
  p <- length(atilde)
  # if (any(diag(B)) == 0) {
  #   cat("Error: exist zero variable", "\n")
  #   msg <- "Error"
  #   break
  # }
  while (loop <= Loop) {
    beta.old <- betainit
    for (j in 1:p){
      a <- diag(B)[j]
      ej <- diag(p)[j,]
      b <- atilde[j] + ej%*%B%*%betainit - a*betainit[j]
      if (j == 1) {
        #no penalization on intercept
        betainit[1] <- -b/a 
      } else {
        betainit[j] <- -soft(b/a,lambda/a)
      }
    }
    # dif <- sum((betainit-beta.old)^2)
    # use the change in loss function to measure 
    dif <- (atilde)%*%(betainit-beta.old) + (betainit)%*%B%*%(betainit)/2 - 
      (beta.old)%*%B%*%(beta.old)/2
    if (abs(dif) < tol) {
     # cat("At INNER loop", loop, "Successful converge", "\n")
      msg <- "Successful convergence"
      break
    }
    if (loop == Loop) {
      cat("INNER loop maximum iteration reached", "\n")
    } 
    loop <- loop + 1
  }
  list(betainit=betainit, message=msg)
}

#loss function: negative log-likelihood function for logistic regression, the input X is a n*d matrix
NLogLik <- function(beta, X, Y){
  design = cbind(1, X)
  -sum(Y*(design%*%t(t(beta))) - log(1 + exp(design%*%t(t(beta)))))/length(Y)
}

#first order gradient of Loss
Lgradient <- function(beta, X, Y){
  design = cbind(1, X)
  -t(Y - expit(design%*%t(t(beta))))%*%design/length(Y)
}

#second-order gradient of Loss
Lgradient2 <- function(beta, X){
  design = cbind(1, X)
  Z = expit(design%*%beta)
  t(c(Z*(1-Z))*design)%*%design/nrow(X)
}


#compare methods: local/global/average/adap1_local/adap1_cv/adap2_local/adap2_cv 
compare.methods <- function(Xall, Yall, site, local_site = c(6, 1), norder = NULL){
  
  #### input:
  # Xall: covariates, a N-by-(p-1) matrix
  # Yall: outcome, a vector of length N, consisting of 0s and 1s
  # site: site, a vector of length N, each site is represented by a number
  # local_site: specify the local site and the site to provide betatilde (optional)
  # norder: assign observations to each fold of 5CV, if NULL then generate it randomly 
  
  ############################################################
  K <- length(unique(site))
  p <- dim(Xall)[2] + 1
  tab <- table(Yall, site)
  nsite <- table(site)
  ############################################################
  
  
  ######################################
  # PART 1.. Local estimator  #
  #####################################
  m <- length(local_site)
  Beta_local <- matrix(NA, nrow = m, ncol = p)
  
  for(i in 1:m){
    Xlocal <- Xall[which(site == local_site[i]),]
    Ylocal <- Yall[which(site == local_site[i])]
    fit0 <- cv.glmnet(Xlocal, Ylocal, family = "binomial")
    Beta_local[i,] <- as.matrix(coef(fit0, s = "lambda.min"))
  }
  
  
  ######################################
  # PART 2.. Pooled estimator  #
  #####################################
  fitall <- cv.glmnet(Xall, Yall, family = "binomial")
  betaall <- as.matrix(coef(fitall, s = "lambda.min"))
  
  
  
  ######################################
  # PART 3.. simple average estimator  #
  #####################################
  Beta <- matrix(0, nrow = K, ncol = p)   # Store the estimator in each site
  WBeta <- nsite/sum(nsite)               # the weight of each site 
  
  # fit logistic regression in each site
  for (i in 1:K){
    Xsite <- Xall[which(site == i),]      # Predictors in the ith site
    Ysite <- Yall[which(site == i)]       # Outcomes in the ith site
    
    tryCatch({
      
      fit.site <- cv.glmnet(Xsite, Ysite, family = "binomial")
      Beta[i,] <- as.matrix(coef(fit.site, s = "lambda.min"))
      
    }, error = function(err) {
      Beta[i,] <- rep(NA, length(betaall))
      cat("At Site", i, "glmnet has error", "\n")
    })
  }
  
  # average the estimates from all sites
  betaAveg <- apply(diag(WBeta)%*%Beta, 2, sum, na.rm = TRUE)  
  
  
  ######################################
  # PART 4.. ODAL estimator  #
  #####################################
  
  # calculate the gradient in each site
  L <- matrix(NA, nrow = K, ncol = p)         # Store the first order gradient (each is a p-dimensional vector)
  L2 <- matrix(NA, nrow = K, ncol = p^2)      # Store the second order gradient (each is a p*p matrix, we expand it into a vector)
  
  # calculate the gradient at betatilde
  Ltilde <- matrix(NA, nrow = m, ncol = p)    # Store the first order gradient (each is a p-dimensional vector)
  L2tilde <- matrix(NA, nrow = m, ncol = p^2) # Store the second order gradient (each is a p*p matrix, we expand it into a vector)
  
  
  # initial value, could be local or the average estimator
  betabar <- betaAveg          # or Beta_local[1,]     
  # the beta to expand the objective function, could be the same as betabar
  betatilde <- Beta_local[2,]   
  
  
  for (i in 1:K){
    Xsite <- Xall[which(site==i),]                   # Predictors in the ith site
    Ysite <- Yall[which(site==i)]                    # Outcomes in the ith site
    L[i,] <- Lgradient(betabar, Xsite, Ysite)        # first order gradient in the ith site
    L2[i,] <- as.vector(Lgradient2(betabar, Xsite))  # second order gradient in the ith site
  }
  
  
  for(i in 1:m){
    Xlocal <- Xall[which(site==local_site[i]),]
    Ylocal <- Yall[which(site==local_site[i])]
    Ltilde[i,] <- Lgradient(betatilde, Xlocal, Ylocal)        # first order gradient in the ith site
    L2tilde[i,] <- as.vector(Lgradient2(betatilde, Xlocal))   # second order gradient in the ith site
  }
  
  
  # Calculate the global first order gradient L
  L_all <- apply(diag(nsite)%*%L, 2, sum)/(sum(nsite))
  # Calculate the global second order gradient L2
  L2_all <- apply(diag(nsite)%*%L2, 2, sum)/(sum(nsite))
  
  
  Beta_est_odal1_5cv <- matrix(NA, nrow = m, ncol = p)
  Beta_est_odal1 <- matrix(NA, nrow = m, ncol = p)
  Beta_est_odal2_5cv <- matrix(NA, nrow = m, ncol = p)
  Beta_est_odal2 <- matrix(NA, nrow = m, ncol = p)
  
  
  m <- 1
  for(i in 1:m){
    n <- as.numeric(nsite[local_site[i]])
    
    Xlocal <- Xall[which(site == local_site[i]), ]
    Ylocal <- Yall[which(site == local_site[i])]
    
    
    # The methods: 1. ADAP1   2. ADAP2
    # Two tuning methods: 1. 5CV 2. local lambda
    
    
    ################################################
    ###  ADAP1  ###
    ################################################
    
    
    # initially use betatilde to get quadratic approximation
    B <- Lgradient2(betatilde, Xlocal)
    atilde <- Lgradient(betatilde, Xlocal, Ylocal) - t(betatilde)%*%B - 
      Lgradient(betabar, Xlocal, Ylocal) + t(L_all)
    
    
    lam.max <- max(abs(atilde + t((B - diag(diag(B)))%*%betabar)))
    lam.min <- ifelse(n < p, 0.02, 1e-04)*lam.max
    lam.seq <- exp(seq(log(lam.min), log(lam.max), length = 100))
    
    
    ##########################
    ###  5CV  ###
    ##########################
    
    
    #if (tune == "5CV"){
    nfold <- 5
    if (is.null(norder))
      norder <- sample(seq_len(n),n)
    
    lam_path <- matrix(ncol = nfold, nrow = length(lam.seq), NA)
    
    
    ndel <- round(n/nfold)
    for (f in seq_len(nfold)){
      if (f != nfold) {
        iddel <- norder[(1 + ndel * (f - 1)):(ndel * f)]
      } else {
        iddel <- norder[(1 + ndel * (f - 1)):n]
      }
      ndel <- length(iddel)
      nf <- n - ndel
      idkeep <- (seq_len(n))[-iddel]
      
      Xf <- Xlocal[-iddel, ]
      Xfdel <- Xlocal[iddel, ]
      Yf <- as.matrix(Ylocal[-iddel])
      Yfdel <- as.matrix(Ylocal[iddel])
      
      # adjust for Cross Validation procedure
      # the global function excludes the validation set 
      Bf <- Lgradient2(betatilde, Xf) 
      atilde_f <- Lgradient(betatilde, Xf, Yf) - t(betatilde)%*%Bf - 
        Lgradient(betabar, Xf, Yf) + (L_all*(sum(nsite)) - 
                                        Lgradient(betabar, Xfdel, Yfdel)*ndel)/(sum(nsite) - ndel)
      betainit <- rep(0, p)
      
      for (la in 1:length(lam.seq)){
        out.loop <- 1
        out.Loop <- 100
        tol <- 1e-4
        msg <- 0
        
        while (out.loop <= out.Loop) {
          fitcd <- coordi(atilde_f, Bf, betainit, lambda = rev(lam.seq)[la])
          beta.new <- fitcd$betainit
          if (is.na(fitcd$message)) {
            msg <- 1
            cat("At OUTER loop", out.loop, "Lambda is too small", "\n")
            break
          }
          #dif <- sum((beta.new-betainit)^2)
          dif <- (atilde_f)%*%(beta.new - betainit) + (beta.new)%*%Bf%*%(beta.new)/2 - 
            (betainit)%*%Bf%*%(betainit)/2
          
          Bf <- Lgradient2(beta.new, Xf)
          atilde_f <- Lgradient(beta.new, Xf, Yf) - t(beta.new)%*%Bf - 
            Lgradient(betabar, Xf, Yf) + (L_all*(sum(nsite)) - 
                                            Lgradient(betabar, Xfdel, Yfdel)*ndel)/(sum(nsite) - ndel)
          betainit <- beta.new
          if (abs(dif) < tol) {
            cat("At OUTER loop", out.loop, "Successful converge", "\n")
            break
          }
          ######### revised 12/30/2020: add break, similar for the following
          if (out.loop == out.Loop) {
            cat("OUTER loop maximum iteration reached", "\n")
            break
          }
          out.loop <- out.loop + 1
        }
        if (msg == 1) break
        if (out.loop < out.Loop) lam_path[la,f] <- NLogLik(beta.new, Xfdel, Yfdel)
      }
    }
    index <- order(colSums(lam_path))
    crerr <- rowSums(lam_path[, index])/length(index) * nfold
    lam.est_odal1_5cv <- lam.est <- rev(lam.seq)[which.min(crerr)]
    
    #fit the final estimator
    betainit <- betatilde
    B <- Lgradient2(betatilde, Xlocal)
    atilde <- Lgradient(betatilde, Xlocal, Ylocal) - t(betatilde)%*%B - 
      Lgradient(betabar, Xlocal, Ylocal) + t(L_all)
    tol <- 1e-5
    out.loop <- 1
    out.Loop <- 100
    msg <- 0
    while (out.loop <= out.Loop) {
      fitcd <- coordi(atilde, B, betainit, lambda = lam.est)
      if (is.na(fitcd$message)){
        cat("The local lambda is too small", "\n")
        msg <- 1
        break
      }
      beta.new <- fitcd$betainit
      #dif <- sum((beta.new-betainit)^2)
      dif <- (atilde)%*%(beta.new - betainit) + (beta.new)%*%B%*%(beta.new)/2 - 
        (betainit)%*%B%*%(betainit)/2
      
      B <- Lgradient2(beta.new, Xlocal)
      atilde <- Lgradient(beta.new, Xlocal, Ylocal) - t(beta.new)%*%B - 
        Lgradient(betabar, Xlocal, Ylocal) + t(L_all)
      betainit <- beta.new
      if (abs(dif) < tol) {
        cat("At OUTER loop", out.loop, "Successful converge", "\n")
        break
      }
      if (out.loop == out.Loop) {
        cat("OUTER loop maximum iteration reached", "\n")
        break
      }
      out.loop <- out.loop + 1
    }
    if ((msg != 1)&(out.loop < out.Loop)) Beta_est_odal1_5cv[i,] <- beta.new
    
    
    #} else { 
    
    ##########################
    ###  local lambda  ###
    ##########################
    
    
    #in CV use default measurement: deviance
    lam.est_odal1 <- lam.est <- cv.glmnet(Xlocal, Ylocal, family = "binomial")$lambda.min
    #fit the final estimator
    betainit <- betatilde
    B <- Lgradient2(betatilde, Xlocal)
    atilde <- Lgradient(betatilde, Xlocal, Ylocal) - t(betatilde)%*%B - 
      Lgradient(betabar, Xlocal, Ylocal) + t(L_all)
    tol <- 1e-5
    out.loop <- 1
    out.Loop <- 100
    msg <- 0
    while (out.loop <= out.Loop) {
      fitcd <- coordi(atilde, B, betainit, lambda = lam.est)
      if (is.na(fitcd$message)){
        cat("The local lambda is too small", "\n")
        msg <- 1
        break
      }
      beta.new <- fitcd$betainit
      #dif <- sum((beta.new-betainit)^2)
      dif <- (atilde)%*%(beta.new - betainit) + (beta.new)%*%B%*%(beta.new)/2 - 
        (betainit)%*%B%*%(betainit)/2
      
      B <- Lgradient2(beta.new, Xlocal)
      atilde <- Lgradient(beta.new, Xlocal, Ylocal) - t(beta.new)%*%B - 
        Lgradient(betabar, Xlocal, Ylocal) + t(L_all)
      betainit <- beta.new
      if (abs(dif) < tol) {
        cat("At OUTER loop", out.loop, "Successful converge", "\n")
        break
      }
      if (out.loop == out.Loop) {
        cat("OUTER loop maximum iteration reached", "\n")
        break
      }
      out.loop <- out.loop + 1
    }
    if ((msg != 1)&(out.loop < out.Loop)) Beta_est_odal1[i,] <- beta.new
    #}
    
    #}else{
    
    
    ################################################
    ###  ADAP2  ###
    ################################################
    
    
    #use betatilde to get quadratic approximation
    B <- Lgradient2(betatilde, Xlocal) + matrix(L2_all, ncol = p, nrow = p) - 
      Lgradient2(betabar, Xlocal) 
    atilde <- Lgradient(betatilde, Xlocal, Ylocal) - t(betatilde)%*%Lgradient2(betatilde, Xlocal) + 
      t(L_all) - Lgradient(betabar, Xlocal, Ylocal) - 
      t(betabar)%*%(matrix(L2_all, ncol = p, nrow = p) - Lgradient2(betabar, Xlocal))
    
    
    lam.max <- max(abs(atilde + t((B - diag(diag(B)))%*%betabar)))
    lam.min <- ifelse(n < p, 0.02, 1e-04)*lam.max
    lam.seq <- exp(seq(log(lam.min), log(lam.max), length = 100))
    #lam.seq <- rev(cv.glmnet(Xlocal, Ylocal, family = "binomial", nfolds = 5)$lambda)
    
    
    ##########################
    ###  5CV  ###
    ##########################
    
    
    
    
    #if (tune == "5CV"){
    nfold <- 5
    if (is.null(norder))
      norder <- sample(seq_len(n),n)
    
    lam_path <- matrix(ncol = nfold, nrow = length(lam.seq), NA)
    
    
    ndel <- round(n/nfold)
    for (f in seq_len(nfold)){
      if (f != nfold) {
        iddel <- norder[(1 + ndel * (f - 1)):(ndel * f)]
      } else {
        iddel <- norder[(1 + ndel * (f - 1)):n]
      }
      ndel <- length(iddel)
      nf <- n - ndel
      idkeep <- (seq_len(n))[-iddel]
      
      Xf <- Xlocal[-iddel, ]
      Xfdel <- Xlocal[iddel, ]
      Yf <- as.matrix(Ylocal[-iddel])
      Yfdel <- as.matrix(Ylocal[iddel])
      
      # adjust for Cross Validation procedure
      # the global function excludes the validation set 
      L2_allf <- (L2_all*(sum(nsite)) - as.vector(Lgradient2(betabar, Xfdel))*ndel)/(sum(nsite) - ndel)
      Bf <- Lgradient2(betatilde, Xf) + matrix(L2_allf, ncol = p, nrow = p) - 
        Lgradient2(betabar, Xf) 
      atilde_f <- Lgradient(betatilde, Xf, Yf) - t(betatilde)%*%Lgradient2(betatilde, Xf) + 
        (L_all*(sum(nsite)) - Lgradient(betabar, Xfdel, Yfdel)*ndel)/(sum(nsite) - ndel) - 
        Lgradient(betabar, Xf, Yf) - 
        t(betabar)%*%(matrix(L2_allf, ncol = p, nrow = p) - Lgradient2(betabar, Xf))
      betainit <- rep(0, p)
      
      for (la in 1:length(lam.seq)){
        tol <- 1e-5
        out.loop <- 1
        out.Loop <- 100
        msg <- 0
        
        while (out.loop <= out.Loop) {
          fitcd <- coordi(atilde_f, Bf, betainit, lambda = rev(lam.seq)[la])
          beta.new <- fitcd$betainit
          
          if (is.na(fitcd$message)){
            msg <- 1
            cat("At OUTER loop", out.loop, "Lambda is too small", "\n")
            break
          }
          #dif <- sum((beta.new-betainit)^2)
          dif <- (atilde_f)%*%(beta.new - betainit) + (beta.new)%*%Bf%*%(beta.new)/2 - 
            (betainit)%*%Bf%*%(betainit)/2
          
          
          Bf <- Lgradient2(beta.new, Xf) + matrix(L2_allf, ncol = p, nrow = p) - 
            Lgradient2(betabar, Xf) 
          atilde_f <- Lgradient(beta.new, Xf, Yf) - t(beta.new)%*%Lgradient2(beta.new, Xf) + 
            (L_all*(sum(nsite)) - Lgradient(betabar, Xfdel, Yfdel)*ndel)/(sum(nsite) - ndel) - 
            Lgradient(betabar, Xf, Yf) - 
            t(betabar)%*%(matrix(L2_allf, ncol = p, nrow = p) - Lgradient2(betabar, Xf))
          
          betainit <- beta.new
          if (abs(dif) < tol) {
            cat("At OUTER loop", out.loop, "Successful converge", "\n")
            break
          }
          if (out.loop == out.Loop) {
            cat("OUTER loop maximum iteration reached", "\n")
            break
          } 
          out.loop <- out.loop + 1
        }
        if (msg==1) break
        if (out.loop < out.Loop) lam_path[la,f] <- NLogLik(beta.new, Xfdel, Yfdel)
      }
    }
    index <- order(colSums(lam_path))
    crerr <- rowSums(lam_path[, index])/length(index) * nfold
    lam.est_odal2_5cv <- lam.est <- rev(lam.seq)[which.min(crerr)]
    
    #fit the final estimator
    betainit <- betatilde
    tol <- 1e-5
    out.loop <- 1
    out.Loop <- 100
    msg <- 0
    while (out.loop <= out.Loop) {
      fitcd <- coordi(atilde, B, betainit, lambda = lam.est)
      if (is.na(fitcd$message)){
        cat("The local lambda is too small", "\n")
        msg <- 1
        break
      }
      beta.new <- fitcd$betainit
      
      #dif <- sum((beta.new-betainit)^2)
      dif <- (atilde)%*%(beta.new - betainit) + (beta.new)%*%B%*%(beta.new)/2 - 
        (betainit)%*%B%*%(betainit)/2
      B <- Lgradient2(beta.new, Xlocal) + matrix(L2_all, ncol = p, nrow = p) - 
        Lgradient2(betabar, Xlocal) 
      atilde <- Lgradient(beta.new, Xlocal, Ylocal) - t(beta.new)%*%Lgradient2(beta.new, Xlocal) + 
        t(L_all) - Lgradient(betabar, Xlocal, Ylocal) - 
        t(betabar)%*%(matrix(L2_all, ncol = p, nrow = p) - Lgradient2(betabar, Xlocal))
      
      betainit <- beta.new
      if (abs(dif) < tol) {
        cat("At OUTER loop", out.loop, "Successful converge", "\n")
        break
      }
      if (out.loop == out.Loop) {
        cat("OUTER loop maximum iteration reached", "\n")
        break
      }
      out.loop <- out.loop + 1
    }
    if ((msg != 1)&(out.loop < out.Loop)) Beta_est_odal2_5cv[i,] <- beta.new
    
    
    #} else {
    
    
    ##########################
    ###  local lambda  ###
    ##########################
    
    
    
    #in CV use default measurement: deviance
    lam.est_odal2 <- lam.est <- cv.glmnet(Xlocal, Ylocal, family = "binomial")$lambda.min
    
    #fit the final estimator
    betainit <- betatilde
    tol <- 1e-5
    out.loop <- 1
    out.Loop <- 100
    msg <- 0
    while (out.loop <= out.Loop) {
      fitcd <- coordi(atilde, B, betainit, lambda = lam.est)
      if (is.na(fitcd$message)){
        cat("The local lambda is too small", "\n")
        msg <- 1
        break
      }
      beta.new <- fitcd$betainit
      
      #dif <- sum((beta.new-betainit)^2)
      dif <- (atilde)%*%(beta.new-betainit) + (beta.new)%*%B%*%(beta.new)/2 - 
        (betainit)%*%B%*%(betainit)/2
      B <- Lgradient2(beta.new, Xlocal) + matrix(L2_all, ncol = p, nrow = p) - 
        Lgradient2(betabar, Xlocal) 
      atilde <- Lgradient(beta.new, Xlocal, Ylocal) - t(beta.new)%*%Lgradient2(beta.new, Xlocal) + 
        t(L_all) - Lgradient(betabar, Xlocal, Ylocal) - 
        t(betabar)%*%(matrix(L2_all, ncol = p, nrow = p) - Lgradient2(betabar, Xlocal))
      
      betainit <- beta.new
      if (abs(dif) < tol) {
        cat("At OUTER loop", out.loop, "Successful converge", "\n")
        break
      }
      if (out.loop == out.Loop) {
        cat("OUTER loop maximum iteration reached", "\n")
        break
      }
      out.loop <- out.loop + 1
    }
    if ((msg != 1)&(out.loop < out.Loop)) Beta_est_odal2[i,] <- beta.new
    
  }
  
  
  
  ######################################
  # PART 5.. output  #
  #####################################
  
  return(output = list(estimation_local = Beta_local[1,], 
                       estimation_pooled = betaall,
                       estimation_ave = betaAveg,
                       
                       estimation_odal1_5cv = Beta_est_odal1_5cv[1,],
                       estimation_odal1 = Beta_est_odal1[1,],
                       
                       estimation_odal2_5cv = Beta_est_odal2_5cv[1,],
                       estimation_odal2 = Beta_est_odal2[1,],
                       
                       lam.est_odal1_5cv = lam.est_odal1_5cv,
                       lam.est_odal1 = lam.est_odal1,
                       
                       lam.est_odal2_5cv = lam.est_odal2_5cv,
                       lam.est_odal2 = lam.est_odal2))
                       
}

