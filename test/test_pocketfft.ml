open Pocketfft
open Bigarray

let epsilon = 1e-10

let epsilon_f32 = 1e-5

let complex_equal ~epsilon c1 c2 =
  abs_float Complex.(c1.re -. c2.re) < epsilon && abs_float Complex.(c1.im -. c2.im) < epsilon

let float_equal ~epsilon f1 f2 = abs_float (f1 -. f2) < epsilon

let create_test_signal_1d n =
  let data = Array1.create complex64 c_layout n in
  for i = 0 to n - 1 do
    let t = float_of_int i /. float_of_int n in
    let re = cos (2.0 *. Float.pi *. 3.0 *. t) +. (0.5 *. cos (2.0 *. Float.pi *. 7.0 *. t)) in
    let im = sin (2.0 *. Float.pi *. 3.0 *. t) +. (0.5 *. sin (2.0 *. Float.pi *. 7.0 *. t)) in
    data.{i} <- Complex.{re; im}
  done ;
  data

let create_test_signal_1d_f32 n =
  let data = Array1.create complex32 c_layout n in
  for i = 0 to n - 1 do
    let t = float_of_int i /. float_of_int n in
    let re = cos (2.0 *. Float.pi *. 3.0 *. t) +. (0.5 *. cos (2.0 *. Float.pi *. 7.0 *. t)) in
    let im = sin (2.0 *. Float.pi *. 3.0 *. t) +. (0.5 *. sin (2.0 *. Float.pi *. 7.0 *. t)) in
    data.{i} <- Complex.{re; im}
  done ;
  data

let create_real_signal_1d n =
  let data = Array1.create float64 c_layout n in
  for i = 0 to n - 1 do
    let t = float_of_int i /. float_of_int n in
    data.{i} <- cos (2.0 *. Float.pi *. 3.0 *. t) +. (0.5 *. cos (2.0 *. Float.pi *. 7.0 *. t))
  done ;
  data

(* Test C2C FFT Float64 *)
let test_c2c_f64_forward_inverse () =
  let n = 64 in
  let input = create_test_signal_1d n in
  let forward = Array1.create complex64 c_layout n in
  let inverse = Array1.create complex64 c_layout n in
  c2c_f64 ~shape:[|n|] ~stride_in:[|16|] ~stride_out:[|16|] ~axes:[|0|] ~forward:true ~fct:1.0
    ~data_in:input ~data_out:forward ~nthreads:1 ;
  c2c_f64 ~shape:[|n|] ~stride_in:[|16|] ~stride_out:[|16|] ~axes:[|0|] ~forward:false
    ~fct:(1.0 /. float_of_int n)
    ~data_in:forward ~data_out:inverse ~nthreads:1 ;
  for i = 0 to n - 1 do
    Alcotest.(check bool) "c2c_f64_roundtrip" true (complex_equal ~epsilon input.{i} inverse.{i})
  done

(* Test C2C FFT Float32 *)
let test_c2c_f32_forward_inverse () =
  let n = 32 in
  let input = create_test_signal_1d_f32 n in
  let forward = Array1.create complex32 c_layout n in
  let inverse = Array1.create complex32 c_layout n in
  c2c_f32 ~shape:[|n|] ~stride_in:[|8|] ~stride_out:[|8|] ~axes:[|0|] ~forward:true ~fct:1.0
    ~data_in:input ~data_out:forward ~nthreads:1 ;
  c2c_f32 ~shape:[|n|] ~stride_in:[|8|] ~stride_out:[|8|] ~axes:[|0|] ~forward:false
    ~fct:(1.0 /. float_of_int n)
    ~data_in:forward ~data_out:inverse ~nthreads:1 ;
  for i = 0 to n - 1 do
    Alcotest.(check bool)
      "c2c_f32_roundtrip" true
      (complex_equal ~epsilon:epsilon_f32 input.{i} inverse.{i})
  done

(* Test R2C FFT *)
let test_r2c_f64 () =
  let n = 64 in
  let input = create_real_signal_1d n in
  let output = Array1.create complex64 c_layout ((n / 2) + 1) in
  r2c_f64 ~shape_in:[|n|] ~stride_in:[|8|] ~stride_out:[|16|] ~axes:[|0|] ~forward:true ~fct:1.0
    ~data_in:input ~data_out:output ~nthreads:1 ;
  Alcotest.(check bool) "r2c_dc_real" true (abs_float output.{0}.im < epsilon) ;
  Alcotest.(check int) "r2c_output_size" ((n / 2) + 1) (Array1.dim output)

(* Test C2R FFT *)
let test_c2r_f64 () =
  let n = 64 in
  let real_input = create_real_signal_1d n in
  let complex_temp = Array1.create complex64 c_layout ((n / 2) + 1) in
  let real_output = Array1.create float64 c_layout n in
  r2c_f64 ~shape_in:[|n|] ~stride_in:[|8|] ~stride_out:[|16|] ~axes:[|0|] ~forward:true ~fct:1.0
    ~data_in:real_input ~data_out:complex_temp ~nthreads:1 ;
  c2r_f64 ~shape_out:[|n|] ~stride_in:[|16|] ~stride_out:[|8|] ~axes:[|0|] ~forward:false
    ~fct:(1.0 /. float_of_int n)
    ~data_in:complex_temp ~data_out:real_output ~nthreads:1 ;
  for i = 0 to n - 1 do
    Alcotest.(check bool) "c2r_roundtrip" true (float_equal ~epsilon real_input.{i} real_output.{i})
  done

(* Test DCT *)
let test_dct_f64 () =
  let n = 32 in
  let input = create_real_signal_1d n in
  let output = Array1.create float64 c_layout n in
  dct_f64 ~shape:[|n|] ~stride_in:[|8|] ~stride_out:[|8|] ~axes:[|0|] ~dct_type:2 ~ortho:false
    ~fct:1.0 ~data_in:input ~data_out:output ~nthreads:1 ;
  let different = ref false in
  for i = 0 to n - 1 do
    if abs_float (input.{i} -. output.{i}) > epsilon then different := true
  done ;
  Alcotest.(check bool) "dct_transforms_data" true !different

(* Test DST *)
let test_dst_f64 () =
  let n = 32 in
  let input = create_real_signal_1d n in
  let output = Array1.create float64 c_layout n in
  dst_f64 ~shape:[|n|] ~stride_in:[|8|] ~stride_out:[|8|] ~axes:[|0|] ~dct_type:2 ~ortho:false
    ~fct:1.0 ~data_in:input ~data_out:output ~nthreads:1 ;
  let different = ref false in
  for i = 0 to n - 1 do
    if abs_float (input.{i} -. output.{i}) > epsilon then different := true
  done ;
  Alcotest.(check bool) "dst_transforms_data" true !different

let test_multithreading () =
  let n = 256 in
  let input = create_test_signal_1d n in
  let output1 = Array1.create complex64 c_layout n in
  let output2 = Array1.create complex64 c_layout n in
  c2c_f64 ~shape:[|n|] ~stride_in:[|16|] ~stride_out:[|16|] ~axes:[|0|] ~forward:true ~fct:1.0
    ~data_in:input ~data_out:output1 ~nthreads:1 ;
  c2c_f64 ~shape:[|n|] ~stride_in:[|16|] ~stride_out:[|16|] ~axes:[|0|] ~forward:true ~fct:1.0
    ~data_in:input ~data_out:output2 ~nthreads:4 ;
  for i = 0 to n - 1 do
    Alcotest.(check bool)
      "multithreading_consistency" true
      (complex_equal ~epsilon output1.{i} output2.{i})
  done

let test_single_element () =
  let input = Array1.create complex64 c_layout 1 in
  let output = Array1.create complex64 c_layout 1 in
  input.{0} <- Complex.{re= 42.0; im= 13.0} ;
  c2c_f64 ~shape:[|1|] ~stride_in:[|16|] ~stride_out:[|16|] ~axes:[|0|] ~forward:true ~fct:1.0
    ~data_in:input ~data_out:output ~nthreads:1 ;
  Alcotest.(check bool) "single_element" true (complex_equal ~epsilon input.{0} output.{0})

let test_power_of_two_sizes () =
  let sizes = [2; 4; 8; 16; 32; 64; 128; 256] in
  List.iter
    (fun n ->
      let input = create_test_signal_1d n in
      let forward = Array1.create complex64 c_layout n in
      let inverse = Array1.create complex64 c_layout n in
      c2c_f64 ~shape:[|n|] ~stride_in:[|16|] ~stride_out:[|16|] ~axes:[|0|] ~forward:true ~fct:1.0
        ~data_in:input ~data_out:forward ~nthreads:1 ;
      c2c_f64 ~shape:[|n|] ~stride_in:[|16|] ~stride_out:[|16|] ~axes:[|0|] ~forward:false
        ~fct:(1.0 /. float_of_int n)
        ~data_in:forward ~data_out:inverse ~nthreads:1 ;
      for i = 0 to n - 1 do
        Alcotest.(check bool)
          (Printf.sprintf "power_of_two_%d" n)
          true
          (complex_equal ~epsilon input.{i} inverse.{i})
      done )
    sizes

let test_non_power_of_two_sizes () =
  let sizes = [3; 5; 7; 9; 15; 17; 31; 63] in
  List.iter
    (fun n ->
      let input = create_test_signal_1d n in
      let forward = Array1.create complex64 c_layout n in
      let inverse = Array1.create complex64 c_layout n in
      c2c_f64 ~shape:[|n|] ~stride_in:[|16|] ~stride_out:[|16|] ~axes:[|0|] ~forward:true ~fct:1.0
        ~data_in:input ~data_out:forward ~nthreads:1 ;
      c2c_f64 ~shape:[|n|] ~stride_in:[|16|] ~stride_out:[|16|] ~axes:[|0|] ~forward:false
        ~fct:(1.0 /. float_of_int n)
        ~data_in:forward ~data_out:inverse ~nthreads:1 ;
      for i = 0 to n - 1 do
        Alcotest.(check bool)
          (Printf.sprintf "non_power_of_two_%d" n)
          true
          (complex_equal ~epsilon input.{i} inverse.{i})
      done )
    sizes

let () =
  let open Alcotest in
  run "PocketFFT"
    [ ( "PocketFFT :: Functions tests"
      , [ test_case "C2C F64 forward/inverse" `Quick test_c2c_f64_forward_inverse
        ; test_case "C2C F32 forward/inverse" `Quick test_c2c_f32_forward_inverse
        ; test_case "R2C F64" `Quick test_r2c_f64
        ; test_case "C2R F64" `Quick test_c2r_f64
        ; test_case "DCT F64" `Quick test_dct_f64
        ; test_case "DST F64" `Quick test_dst_f64 ] )
    ; ("PocketFFT :: Multithreading", [test_case "Consistency" `Quick test_multithreading])
    ; ( "PocketFFT :: Edge Cases"
      , [ test_case "Single element" `Quick test_single_element
        ; test_case "Power of two sizes" `Quick test_power_of_two_sizes
        ; test_case "Non-power of two sizes" `Quick test_non_power_of_two_sizes ] ) ]
