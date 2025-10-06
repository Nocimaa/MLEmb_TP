
//Question 1
float linear_regression_prediction(float* features, float* thetas, int n_parameters)
{
    float res = thetas[0];
    for (int i = 0; i < n_parameters; i++)
    {
        res += features[i] * thetas[i + 1];
    }
    return res;
}

//Question 2
float puissance(float x, int n) {
    float result = 1.0;
    for (int i = 0; i < n; i++) {
        result *= x;
    }
    return result;
}

int facto(int n)
{
    if (n == 0 || n == 1)
    {
        return 1;
    }
    else
    {
        int res = 1;
        int i = 0;
        while (i < n)
        {
            i++;
            res *= i;
        }
        return res;
    }
}

float exp_approx(float x, int n_term)
{
    float res = 1;
    for (int i = 1; i <= n_term; i++)
    {
        res += puissance(x, i) / facto(i);
    }
    return res;
}

//Question 3


