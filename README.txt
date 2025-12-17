To run this program, you will need to download the most current version of Python. This can be done by going to the python website and
downloading the software onto your computer following their directions. You must also have Visual Studios Code downloaded to your laptop. 
To do this go to the Visual Studio Code website and download the package for your computer. Follow their instructions for setup. After
downloading Visual Studios Code, you can download the python extension in Visual Studios to run the program from Visual Studios, as opposed 
to running it on your computer's terminal. 

Environment setup:
This project requires multiple python libraries to properly run. The best way to set up these libraries is to create a python environment for
this specific project. Here are the steps to set up your environment using venv:
1. Open the terminal in your computer and navigate to your project's directory before you begin setting up your environment
2. Once you have located your projects directory run the command pip install virtualenv in your terminal
3. Then run the command python<version> -m venv <virtual-environment-name>
4. Now that you have created your virtual environment, you must activate it by running the command, source env/bin/activate
5. You will need to download the packages/libraries into your venv. For this project you will need Pandas, numpy, sklearn, and scipy, so you will run the commands:
       pip install pandas
       pip install numpy
       pip install scikit-learn
       pip install scipy

After this your environment should be ready to use.


Typescript commands:
To save the script of your programs run (as seen in the file 25_RPM_Results) 

script name_of_script
.
.
.
run program
.
.
.
exit

table for faults:

1.0 - Bearing (1) Ball & Bearing (2) Combination 
2.0 - Bearing (1) Ball & Bearing (2) Inner
3.0 - Bearing (1) Ball & Bearing (2) Outer
4.0 - Bearing (1) Ball & Shaft Fault (Centerally bent)
5.0 - Bearing (1) Ball & Shaft Fault (Coupling End bent)
6.0 - Bearing (1) Combination & Bearing (2) Ball
7.0 - Bearing (1) Combination & Bearing (2) Inner
8.0 - Bearing (1) Combination & Bearing (2) Outer
9.0 - Bearing (1) Combination & Shaft Fault (Centerally bent)
10.0 - Bearing (1) Combination & Shaft Fault (Coupling End bent)
11.0 - Bearing (1) Fault (ball)
12.0 - Bearing (1) Fault (combination)
13.0 - Bearing (1) Fault (inner race)
14.0 - Bearing (1) Fault (outer race)
15.0 - Bearing (1) Inner & Bearing (2) Ball
16.0 - Bearing (1) Inner & Bearing (2) Combintion
17.0 - Bearing (1) Inner & Bearing (2) Outer
18.0 - Bearing (1) Inner & Shaft Fault ( Coupling End Bent)
19.0 - Bearing (1) Inner & Shaft Fault (Centerally bent)
20.0 - Bearing (1) Outer & Bearing (2) Ball
21.0 - Bearing (1) Outer & Bearing (2) Combination
22.0 - Bearing (1) Outer & Bearing (2) Inner
23.0 - Bearing (1) Outer & Shaft Fault (Centerally bent)
24.0 - Bearing (1) Outer & Shaft Fault (Coupling End bent)
25.0 - Bearing (2) Ball & Shaft Fault (Centerally bent)
26.0 - Bearing (2) Ball & Shaft Fault (Coupling End bent)
27.0 - Bearing (2) Combination & Shaft Fault (Centerally bent)
28.0 - Bearing (2) Combination & Shaft Fault (Coupling End bent)
29.0 - Bearing (2) Fault (ball)
30.0 - Bearing (2) Fault (combination)
31.0 - Bearing (2) Fault (inner race)
32.0 - Bearing (2) Fault (outer race)
33.0 - Bearing (2) Inner & Shaft Fault (Centerally bent)
34.0 - Bearing (2) Inner & Shaft Fault (Coupling End Bent)
35.0 - Bearing (2) Outer & Shaft Fault (Centerally bent)
36.0 - Bearing (2) Outer & Shaft Fault (Coupling End bent)
37.0 - No Fault
38.0 - Shaft Fault (Centerally bent) 
39.0 - Shaft Fault (Coupling end bent)

Extra Notes:
When running the Frequency_Domain_FE make sure your computers path to the CSV files matches the path in the program.




