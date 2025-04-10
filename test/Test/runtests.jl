# Copyright (c) 2017: Miles Lubin and contributors
# Copyright (c) 2017: Google Inc.
#
# Use of this source code is governed by an MIT-style license that can be found
# in the LICENSE.md file or at https://opensource.org/licenses/MIT.

module TestTest

using Test
import MathOptInterface as MOI
import LowRankOpt as LRO

function runtests()
    for name in names(@__MODULE__; all = true)
        if startswith("$name", "test_")
            @testset "$(name)" begin
                getfield(@__MODULE__, name)()
            end
        end
    end
    return
end

function test_runtests()
    # Some tests are excluded because UniversalFallback accepts absolutely
    # everything.
    MOI.Test.runtests(
        MOI.Utilities.MockOptimizer(
            MOI.Utilities.UniversalFallback(MOI.Utilities.Model{Float64}()),
        ),
        MOI.Test.Config(),
        warn_unsupported = true,
        verbose = true,
        test_module = LRO.Test,
    )
    return
end

end  # module

TestTest.runtests()
