(lang dune 3.20)

(name pocketfft)

(generate_opam_files true)

(source
 (github gabyfle/ocaml-pocketfft))

(authors "Gabriel Santamaria <gaby.santamaria@outlook.fr>")

(maintainers "Gabriel Santamaria <gaby.santamaria@outlook.fr>")

(license LICENSE)

(package
 (name pocketfft)
 (synopsis "OCaml bindings for PocketFFT")
 (description "OCaml bindings for PocketFFT")
 (depends
  ocaml
  (alcotest :with-test))
 (tags
  ("fft" "bindings")))
