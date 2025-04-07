
test_inputs_dir := "test-inputs"
test_outputs_dir := "test-outputs"
test_inputs_tar := "test-inputs.tar.gz"
boom_modules_dir   := test_inputs_dir + "/" + "boom-modules"
rocket_modules_dir := test_inputs_dir + "/" + "rocket-modules"

[group: 'test']
uncompress:
  tar -xvzf {{test_inputs_tar}}
  python scripts/firrtl-module-splitter.py {{test_inputs_dir}}/chipyard.harness.TestHarness.LargeBoomV3Config.fir  {{boom_modules_dir}}
  python scripts/firrtl-module-splitter.py {{test_inputs_dir}}/chipyard.harness.TestHarness.RocketConfig.fir       {{rocket_modules_dir}}

[group: 'test']
make_output_dir:
  mkdir -p {{test_outputs_dir}}

[group: 'test']
test: uncompress make_output_dir
  cargo nextest run --release --no-fail-fast

[group: 'test']
test_debug: uncompress make_output_dir
  cargo nextest run

[group: 'test']
test_only name:
  RUST_BACKTRACE=full cargo nextest run --release {{name}} --nocapture

[group: 'test']
test_only_debug name:
  RUST_BACKTRACE=full cargo nextest run {{name}} --nocapture

[group: 'test']
list:
  cargo nextest list

[group: 'test']
repackage_test_inputs:
  rm -rf {{boom_modules_dir}}
  rm -rf {{rocket_modules_dir}}
  rm {{test_inputs_tar}}
  tar -cvzf {{test_inputs_tar}} {{test_inputs_dir}}

[group: 'clean']
clean:
  rm -rf {{test_inputs_dir}} {{test_outputs_dir}}

[group: 'clean']
clean_build:
  cargo clean

[group: 'clean']
clean_all: clean clean_build

[group: 'firtool']
firtool:
  firtool \
      --format=fir \
      --export-module-hierarchy \
      --verify-each=true \
      --warn-on-unprocessed-annotations \
      --disable-annotation-classless \
      --disable-annotation-unknown \
      --mlir-timing \
      --lowering-options=emittedLineLength=2048,noAlwaysComb,disallowLocalVariables,verifLabels,disallowPortDeclSharing,locationInfoStyle=wrapInAtSquareBracket \
      --repl-seq-mem \
      --split-verilog \
      -o test-outputs/verilog \
      ./test-outputs/chipyard.harness.TestHarness.RocketConfig.fir
# ./test-outputs/GCD.fir

# --repl-seq-mem-file=$(MFC_SMEMS_CONF) \
# --annotation-file=$(FINAL_ANNO_FILE) \
