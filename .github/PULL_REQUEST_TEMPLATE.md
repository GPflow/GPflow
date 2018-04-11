# Standard GPflow pull request

At first, thank you very much for spending time on contibuting to GPflow.\
This template exists to help us to grasp your work without asking very common and questions. The template helps you to get thorough feedback as well.


## PR content:

* [Title](#title)
* [Description](#description)
* [Minimal working example](#minimal-working-example)

### Title

Squeeze it. Be creative. There is no necessity in any marks like [BUG], {feature} and cetera. We have github labels for that.

### Description

Describe your contribution via writing down context and reasons. Make it clear either it is a **bug fix**, **feature** or **enhancement**. Do not make your it very wordy, keep it short and simple.

* Bad example: `This is a bug-fix found at #111.`
* Bad example: `Add new SuperMagic optimizer.`
* Good example: `Working on feature #111, we found that parameters of VGP are not instantiated properly. It causes arbitrary failures at training. The bug is fixed by adding necessary parameters to the list of trainables.`


### Minimal working example

No matter how good description is, the best thing you can do for making reviewer's life much easier is to add a minimal working example. Ideally, it should be a short code snippet with relevant comments. Once, MWE becomes big (> 50 lines), you should consider [gist](https://gist.github.com) instead.

* If your PR is a *bug* related fix, then your MWE must disclose the bug using synthetic example or "real" one.
* When you propose new *feature*, the MWE must show different use cases and clear up benefits from it.

## Code quality requirements

For details, look at [contributing.md](../contributing.md) code quality requirements section. Nevertheless, *Main* points listed here:

* _New feature's_ code must be accompanied with [detailed](../contributing.md) docstrings.
* _New features_ must be covered with tests.
* _Fixed bug_ must be proved by test suites.
* Use type annotations.
* Use `.pylintrc` for code formatting.