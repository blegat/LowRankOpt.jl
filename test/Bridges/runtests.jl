# Copyright (c) 2017: Miles Lubin and contributors
# Copyright (c) 2017: Google Inc.
#
# Use of this source code is governed by an MIT-style license that can be found
# in the LICENSE.md file or at https://opensource.org/licenses/MIT.

using Test

@testset "$(dir)" for dir in ["Variable"]
    @testset "$(file)" for file in readdir(joinpath(@__DIR__, dir))
        if !endswith(file, ".jl")
            continue
        end
        include(joinpath(@__DIR__, dir, file))
    end
end
