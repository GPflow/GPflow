# Standard GPflow pull request

At first, thank you very much for spending time on contributing to GPflow.\
This template exists to help us to grasp your work without asking very common questions. The template helps you to get thorough feedback as well.


## PR content:

* [Title](#title)
* [Description](#description)
* [Minimal working example](#minimal-working-example)

### Title

Squeeze it. Be creative. There is no necessity for any marks like [BUG], {feature} and cetera. We have github labels for that.

### Description

Describe your contribution by writing down the context and reasons. Make it clear whether it is a **bug fix**, **feature** or **enhancement**. Do not make it very wordy, keep it short and simple.

* Bad example: `This is a bug-fix found at #111.`
* Bad example: `Add new SuperMagic optimizer.`
* Good example: `Working on feature #111, we found that parameters of VGP are not instantiated properly. It causes arbitrary failures at training. The bug is fixed by adding necessary parameters to the list of trainables.`


### Minimal working example

No matter how good the description is, the best thing you can do for making a reviewer's life much easier is to add a minimal working example (MWE). Ideally, it should be a short code snippet with relevant comments. If a MWE is big (> 50 lines), you should consider [gist](https://gist.github.com) instead.

* If your PR is a *bug* related fix, then your MWE must disclose the bug explicitly showing what happened before and after.
* When you propose a new *feature*, the MWE must show different use cases and demonstrate its benefits.

## Code quality requirements

For details, look at the code quality requirements section in [contributing.md](../contributing.md). The *main* points are repeated here:

* _New feature's_ code must be accompanied with [detailed](../contributing.md) docstrings.
* _New features_ must be covered with tests.
* _Fixed bug_ must be proved by test suites.
* Use type annotations.
* Use `.pylintrc` for code formatting.
