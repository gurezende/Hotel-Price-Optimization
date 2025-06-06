# Hotel-Price-Optimization
Hotel Room Price Optimization with PyGAM
![](./blog2.jpeg)

## PyGAM
PyGAM is a Python library with an implementation of GAMs.
To get started, let's install it using pip:
pip install pygam
Next, let's go over our problem.
Problem: Optimizing Hotel Room Prices
Let's imagine we were hired by a hotel to optimize room prices and maximize revenue. 
As we can imagine, no pricing scenario is simple, as several factors influence demand. Some examples we gathered in this example are the day of the week, the season, the room type, how close you are to attractions, customer reviews, and, of course, the price itself.
Our goal now is to find the optimal price for each room type, considering all these factors, to get the most revenue.

Training data ready for modeling. Image by the authorNext, we create and fit the GAM model. s() denotes a smooth function (spline) and f() denotes a categorical feature. We fit the model using the training data.
```python
# Create a GAM model
gam = GAM(f(0) + f(1) + f(2) + s(3) + s(4) + s(5))

# Fit the model to the training data
gam.fit(X_train, y)
```

The optimized price is the following.
**Optimal Price:** $154.08
**Maximum Revenue:** $9404.62

Max revenue at $154.08. Image by the author.Interpreting the Results

By examining the GAM model's components, you can understand how each factor influences the optimal price. 
Change the conditions to gain insights into how to adjust prices based on them. 
The peak of the prices should be during the Summer for this hotel.

## Known Issues
PyGAM's last commit was on Feb, 2024. Since then, there is a compatibility issue for newer versions of numpy/ scipy. <br>
Here is the fix:
* Manually change line 739 of .venv/Lib/site-packages/pygam/pygam.py to Q, R = np.linalg.qr(WB.toarray()).
* Manually change line 82 of .venv/Lib/site-packages/pygam/utils.py to A = A.toarray().
* There may be more places as well, depending on which models you use. The broad fix is just to replace .A with .toarray() everywhere you get that error.
* For more solutions to this error, refer to the issue [#357](https://github.com/dswah/pyGAM/issues/357) in PyGAM's repo.
