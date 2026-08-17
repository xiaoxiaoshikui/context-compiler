# OAuth safety constraints

Authorization codes are single-use credentials. Never automatically replay an OAuth authorization-code exchange after an ambiguous network failure. A replay can violate provider guarantees and may create inconsistent login state.

Safari callback handling must preserve the same single-exchange invariant as every other browser.
