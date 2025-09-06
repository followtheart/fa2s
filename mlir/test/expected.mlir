// Expected MLIR output for input.fa2s
// This shows the generic assembly representation

module {
  func.func private @fa2s_main() {
    fa2s.state "start"
    fa2s.state "loop" {
      fa2s.guard {
        fa2s.input_guard "any"
        fa2s.stack_guard "A" "*"
        fa2s.stack_guard "B" "*"
      }
      fa2s.actions {
        fa2s.read
        fa2s.write "last"
      }
      fa2s.transition "loop"
    }
    fa2s.state "loop" {
      fa2s.guard {
        fa2s.input_guard "eof"
        fa2s.stack_guard "A" "*"
        fa2s.stack_guard "B" "*"
      }
      fa2s.actions {
        fa2s.halt "accept"
      }
    }
    return
  }
}
