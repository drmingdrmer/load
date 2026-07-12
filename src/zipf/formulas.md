# Zipf Sampling Formulas

The mathematical derivations behind the [Zipf](https://en.wikipedia.org/wiki/Zipf%27s_law) sampler. For an overview and usage, see the [module documentation](crate::zipf).

## Mathematical Foundation

The Zipf distribution follows the power law:

$$P(x) = C \cdot x^{-s}$$

Where:
- $C$ is a normalization constant
- $s > 0$ is the shape parameter (higher values create steeper distributions)
- $x$ represents the rank of an item

```text
 700 +--------------------------------------------------------------------+
     |      +      +      +      +      +     +      +      +      +      |
     |       :                                                    +.....+ |
 600 |-+     +                                                          +-|
     |       :                                                            |
     |       +                                                            |
 500 |-+      :                                                         +-|
     |        +                                                           |
     |         +                                                          |
 400 |-+       +                                                        +-|
     |          +                                                         |
 300 |-+         +                                                      +-|
     |           ++                                                       |
     |             +                                                      |
 200 |-+            ++                                                  +-|
     |                +++                                                 |
     |                  ++++                                              |
 100 |-+                    ++++++                                      +-|
     |                           ++++++++++++                             |
     |      +      +      +      +      +    +++++++++++++++++++++++++++++|
   0 +--------------------------------------------------------------------+
     0      10     20     30     40     50    60     70     80     90    100
```

The Zipf distribution has a key property: it's log-log linear. Taking the natural logarithm of both sides:

$$\ln(P(x)) = \ln(C) - s \cdot \ln(x)$$

This means that on a log-log plot, the distribution appears as a straight line with slope $-s$.

```text
   7 +----------------------------------------------------------------------+
     |      +      +      +      +    +  +      +      +      +      +      |
     |                                 +++                          +.....+ |
   6 |-+                                  +++                             +-|
     |                                       ++++                           |
     |                                           ++++                       |
   5 |-+                                            ++++                  +-|
     |                                                 ++++                 |
     |                                                     ++++             |
   4 |-+                                                      ++++        +-|
     |                                                           +++++      |
   3 |-+                                                             ++   +-|
     |                                                                      |
     |                                                                      |
   2 |-+                                                                  +-|
     |                                                                      |
     |                                                                      |
   1 |-+                                                                  +-|
     |                                                                      |
     |      +      +      +      +       +      +      +      +      +      |
   0 +----------------------------------------------------------------------+
     0     0.5     1     1.5     2      2.5     3     3.5     4     4.5     5
```

## Algorithm Implementation

The core algorithm uses the [inverse transform sampling](https://en.wikipedia.org/wiki/Inverse_transform_sampling) method. For the probability density function:

$$f(x) = C \cdot x^{-s}$$

We need to find the inverse of its cumulative distribution function (CDF).

**Step 1: Compute the CDF**

$$F(t) = \int_a^t C x^{-s} dx = C \frac{1}{1-s} \cdot (t^{1-s} - a^{1-s})$$

Let $q = 1-s$, then:

$$F(t) = \frac{C}{q}(t^q - a^q)$$

**Step 2: Normalize the distribution**

For the total probability to equal 1:

$$\int_a^b f(x) dx = 1$$

This gives us:

$$C = \frac{q}{b^q - a^q}$$

**Step 3: Inverse transform**

For a uniform random variable $u \in [0,1)$, solve $F(t) = u$:

$$u = \frac{C}{q}(t^q - a^q)$$

$$t = \left(\frac{q}{C} \cdot u + a^q\right)^{1/q} = ((b^q - a^q) u + a^q)^{1/q}$$

Therefore, given uniformly distributed $u$, the resulting $t$ follows a zipf distribution.

### Special Case: s = 1

When $s = 1$, the general formula breaks down because $q = 1-s = 0$, making the term $1/q$ undefined. This case requires special handling using logarithmic integration.

**Why s = 1 is special:**

The standard CDF formula $F(t) = \frac{C}{q}(t^q - a^q)$ becomes $\frac{C}{0}(t^0 - a^0) = \frac{C}{0} \cdot 0$, which is indeterminate.

**Derivation for s = 1:**

Starting with the PDF: $f(x) = C \cdot x^{-1} = \frac{C}{x}$

The CDF becomes:
$$F(t) = \int_a^t \frac{C}{x} dx = C(\ln t - \ln a) = C \ln\left(\frac{t}{a}\right)$$

**Normalization:**

For the total probability to equal 1:
$$\int_a^b \frac{C}{x} dx = C(\ln b - \ln a) = 1$$

Therefore: $C = \frac{1}{\ln b - \ln a}$

**Inverse transform:**

Setting $F(t) = u$ and solving for $t$:
$$u = \frac{\ln t - \ln a}{\ln b - \ln a} = \frac{\ln(t/a)}{\ln(b/a)}$$

$$\ln(t/a) = u \ln(b/a)$$

$$\frac{t}{a} = \left(\frac{b}{a}\right)^u$$

$$t = a \left(\frac{b}{a}\right)^u$$

This logarithmic form ensures numerical stability and is the classical Zipf distribution used in linguistic analysis and web traffic modeling.
