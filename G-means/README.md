# G-Means Clustering   

## Steps to execute programme: (Windows systems)

### Setup C environment:

We need to have MinGW intalled and integrated in windows.

Please follow this link to setup MinGW : 
[Setup MinGW](http://www.multigesture.net/articles/how-to-install-mingw-msys-and-eclipse-on-windows/)

### Setup Project:

Clone the repository using git (ensure [git](https://git-scm.com/book/en/v2/Getting-Started-Installing-Git) is installed on the system)   
  
```git clone https://github.ncsu.edu/sjgurav/G-Means.git```   
   
Go to the cloned repository. 

``` cd G-Means ``` 

### Compile and Run Program:

 Use **git bash** command Prompt to execute the following commands.
   
 Compile the necessary files:  
 
 ```gcc -std=c99 -c gmeans.c seq_kmeans.c data.c command.c cluster.c ```  
   
 Link the object files to create an exceutable file.  
 
 ```gcc -o mygmeans gmeans.o seq_kmeans.o data.o command.o cluster.o -lm```  

Run the executable file:  
```./mygmeans.exe filepath/filename```

#### Example:
``` ./mygmeans.exe data/gmean3.csv ```

``` ./mygmeans.exe data/gmean4.csv ```

``` ./mygmeans.exe data/ecoli.csv ```

#### Note:
Please make sure filepath is relative to the current folder in case if there is any. If file is directly in current folder, there is no need of filepath.

For above command filenames is:

``` gmean2.csv, gmean3.csv, gmean4.csv, gmean5.csv, ecoli.csv ```


#### Note:
Please note that all files are in data folder, hence filepath is 'data'.

#### To plot the graph to display generated clusters run the following command:

``` Rscript plot_gmeans.R ```

Above command will generate Rplots.pdf file with all visualizations.


   
 
