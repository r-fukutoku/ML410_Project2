Create a new Github page with a presentation on the concepts of Locally Weighted Regression and Random Forest. 

Apply the regression methods to real data sets, such as "Cars" or "Boston Housing Data" where you consider only one input variable 
(the weight of the car for the "Cars" data set and the number of rooms for the "Boston Hausing" data). 
The output varable is the mileage for the "Cars" data set and "cmedv" or the median price for the housing data.

For each method and data set report the crossvalidated mean square error and determine which method is achieveng the better results.
In this paper you should also include theoretical considerations, examples of Python coding and plots. 
The final results should be clearly stated.


## Locally Weighted Regression (Loess)
The main idea of linear regression is the assumption that:

𝑦=𝑋⋅𝛽+𝜎𝜖 

If we pre-multiply this equation with a matrix of weights (the "weights" are on the main diagonal and the rest of the elements are 0) we get:

𝑊(𝑖)𝑦=𝑊(𝑖)𝑋⋅𝛽+𝜎𝑊(𝑖)𝜖 

The distancfe bw two independent observation is the Euclidean distance bw the two represented  𝑝− dimensional vectors. The equation is:

𝑑𝑖𝑠𝑡(𝑣⃗ ,𝑤⃗ )=(𝑣1−𝑊1)2+(𝑣2−𝑊2)2+...(𝑣𝑝−𝑊𝑝)2‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾√ 

We shall have  𝑛  differrent weight vectors because we have  𝑛  different observations.

Important aspect: linear regression can be seen as a linear combination of the observed outputs (values of the dependent variable).

We have:

𝑋𝑇𝑦=𝑋𝑇𝑋𝛽+𝜎𝑋𝑇𝜖 

We solve for  𝛽  (by assuming that  𝑋𝑇𝑋  is invertible):

𝛽=(𝑋𝑇𝑋)−1(𝑋𝑇𝑦)−𝜎(𝑋𝑇𝑋)−1𝑋𝑇𝜖 

We take the expected value of this equation and obtain:

𝛽¯=(𝑋𝑇𝑋)−1(𝑋𝑇𝑦) 

Therefore the predictions we make are:

𝑦̂ =(𝑋𝑇𝑋)−1(𝑋𝑇𝑦) 

The big Idea: the predictions we make are a linear combination of the actual observed values of the dependent valuable!

For locally weighted regression,  𝑦̂   is pbtained as a different linear combination of the values of y.

## Random Forest





## Applications with Real Data
```python
%matplotlib inline
%config InlineBackend.figure_format = 'retina'
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams['figure.dpi'] = 120
```













## Welcome to GitHub Pages

You can use the [editor on GitHub](https://github.com/r-fukutoku/Project2/edit/gh-pages/index.md) to maintain and preview the content for your website in Markdown files.

Whenever you commit to this repository, GitHub Pages will run [Jekyll](https://jekyllrb.com/) to rebuild the pages in your site, from the content in your Markdown files.

### Markdown

Markdown is a lightweight and easy-to-use syntax for styling your writing. It includes conventions for

```markdown
Syntax highlighted code block

# Header 1
## Header 2
### Header 3

- Bulleted
- List

1. Numbered
2. List

**Bold** and _Italic_ and `Code` text

[Link](url) and ![Image](src)
```

For more details see [Basic writing and formatting syntax](https://docs.github.com/en/github/writing-on-github/getting-started-with-writing-and-formatting-on-github/basic-writing-and-formatting-syntax).

### Jekyll Themes

Your Pages site will use the layout and styles from the Jekyll theme you have selected in your [repository settings](https://github.com/r-fukutoku/Project2/settings/pages). The name of this theme is saved in the Jekyll `_config.yml` configuration file.

### Support or Contact

Having trouble with Pages? Check out our [documentation](https://docs.github.com/categories/github-pages-basics/) or [contact support](https://support.github.com/contact) and we’ll help you sort it out.
