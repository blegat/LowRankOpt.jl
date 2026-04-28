# Copyright (c) 2024: Benoît Legat and contributors
#
# Use of this source code is governed by an MIT-style license that can be found
# in the LICENSE.md file or at https://opensource.org/licenses/MIT.

module DualizationExt

import Dualization
import LowRankOpt

Dualization.dual_attribute(attr::LowRankOpt.RawStatus) = attr
Dualization.dual_attribute(attr::LowRankOpt.Solution) = attr

end
