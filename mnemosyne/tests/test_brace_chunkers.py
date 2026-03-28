# Copyright 2026 Cast Rock Innovation L.L.C.
# SPDX-License-Identifier: AGPL-3.0-or-later

"""
Tests for the brace-based multi-language chunkers:
  - BraceDepthScanner (shared)
  - GoChunker
  - CSharpChunker
  - RustChunker
  - JavaChunker
  - Registry dispatch for new languages
"""

import unittest


def _default_config():
    """Return a Config with default settings from a temp directory."""
    import tempfile
    from mnemosyne.config import Config
    return Config(root=tempfile.mkdtemp())


# =========================================================================
# BraceDepthScanner tests
# =========================================================================


class TestBraceDepthScanner(unittest.TestCase):

    def setUp(self):
        from mnemosyne.chunkers.brace_chunker import BraceDepthScanner
        self.scan = BraceDepthScanner.find_block_end

    def test_simple_braces(self):
        src = "{ hello }"
        self.assertEqual(self.scan(src, 0), 8)

    def test_nested_braces(self):
        src = "{ { { inner } mid } }"
        self.assertEqual(self.scan(src, 0), len(src) - 1)

    def test_three_deep_nesting(self):
        src = "{ a { b { c } b } a }"
        result = self.scan(src, 0)
        self.assertEqual(src[result], "}")
        self.assertEqual(result, len(src) - 1)

    def test_braces_in_double_quoted_string(self):
        src = '{ "a { b } c" }'
        self.assertEqual(self.scan(src, 0), len(src) - 1)

    def test_braces_in_single_quoted_string(self):
        src = "{ 'a { b }' }"
        self.assertEqual(self.scan(src, 0), len(src) - 1)

    def test_braces_in_backtick_string(self):
        src = "{ `a { b }` }"
        self.assertEqual(self.scan(src, 0), len(src) - 1)

    def test_braces_in_line_comment(self):
        src = "{\n// this { is a comment\n}"
        self.assertEqual(src[self.scan(src, 0)], "}")

    def test_braces_in_block_comment(self):
        src = "{ /* { nested } comment */ }"
        self.assertEqual(self.scan(src, 0), len(src) - 1)

    def test_escaped_quote_in_string(self):
        src = '{ "escaped \\" { brace" }'
        result = self.scan(src, 0)
        self.assertEqual(src[result], "}")

    def test_unmatched_brace_returns_len(self):
        src = "{ open forever"
        result = self.scan(src, 0)
        self.assertEqual(result, len(src))

    def test_mixed_strings_and_comments(self):
        src = '{ "}" /* } */ // }\n}'
        result = self.scan(src, 0)
        self.assertEqual(src[result], "}")


# =========================================================================
# GoChunker tests
# =========================================================================

GO_SOURCE = '''\
package main

import (
    "fmt"
    "os"
)

const (
    MaxRetries = 3
    Timeout    = 30
)

func main() {
    fmt.Println("hello")
}

type Server struct {
    Host string
    Port int
}

func (s *Server) Start() {
    fmt.Printf("Starting %s:%d\\n", s.Host, s.Port)
}

func (s *Server) Stop() {
    fmt.Println("Stopping")
}

type Logger interface {
    Log(msg string)
    Flush()
}
'''


class TestGoChunker(unittest.TestCase):

    def setUp(self):
        from mnemosyne.chunkers.go_chunker import GoChunker
        self.cfg = _default_config()
        self.chunker = GoChunker(self.cfg)

    def test_produces_non_empty_chunks(self):
        chunks = self.chunker.chunk(GO_SOURCE, "go")
        self.assertGreater(len(chunks), 0)

    def test_function_extracted(self):
        chunks = self.chunker.chunk(GO_SOURCE, "go")
        fn_names = {c.symbol_name for c in chunks if c.chunk_type == "function"}
        self.assertIn("main", fn_names)

    def test_struct_extracted(self):
        chunks = self.chunker.chunk(GO_SOURCE, "go")
        cls_names = {c.symbol_name for c in chunks if c.chunk_type == "class"}
        self.assertIn("Server", cls_names)

    def test_interface_extracted(self):
        chunks = self.chunker.chunk(GO_SOURCE, "go")
        cls_names = {c.symbol_name for c in chunks if c.chunk_type == "class"}
        self.assertIn("Logger", cls_names)

    def test_method_has_receiver_qualified_name(self):
        chunks = self.chunker.chunk(GO_SOURCE, "go")
        fn_names = {c.symbol_name for c in chunks if c.chunk_type == "function"}
        self.assertTrue(
            any("Server.Start" == n for n in fn_names),
            f"Expected 'Server.Start' in {fn_names}",
        )
        self.assertTrue(
            any("Server.Stop" == n for n in fn_names),
            f"Expected 'Server.Stop' in {fn_names}",
        )

    def test_import_block_captured(self):
        chunks = self.chunker.chunk(GO_SOURCE, "go")
        import_chunks = [c for c in chunks if c.chunk_type == "imports"]
        self.assertGreater(len(import_chunks), 0)

    def test_const_block_captured(self):
        chunks = self.chunker.chunk(GO_SOURCE, "go")
        # const blocks are captured as imports type
        import_chunks = [c for c in chunks if c.chunk_type == "imports"]
        combined = "".join(c.content for c in import_chunks)
        self.assertIn("MaxRetries", combined)

    def test_line_ranges_valid(self):
        chunks = self.chunker.chunk(GO_SOURCE, "go")
        for c in chunks:
            self.assertGreaterEqual(c.line_start, 1)
            self.assertGreaterEqual(c.line_end, c.line_start)

    def test_empty_source(self):
        chunks = self.chunker.chunk("  \n  ", "go")
        self.assertEqual(chunks, [])


# =========================================================================
# CSharpChunker tests
# =========================================================================

CSHARP_SOURCE = '''\
using System;
using System.Collections.Generic;

namespace MyApp.Services
{
    public interface IService
    {
        void Execute();
        Task<string> GetName();
    }

    public class ServiceImpl : IService
    {
        private readonly string _name;

        public ServiceImpl(string name)
        {
            _name = name;
        }

        public void Execute()
        {
            Console.WriteLine(_name);
        }

        public async Task<string> GetName()
        {
            return await Task.FromResult(_name);
        }
    }

    public class GenericRepo<T> where T : class
    {
        public List<T> GetAll()
        {
            return new List<T>();
        }
    }
}
'''


class TestCSharpChunker(unittest.TestCase):

    def setUp(self):
        from mnemosyne.chunkers.csharp_chunker import CSharpChunker
        self.cfg = _default_config()
        self.chunker = CSharpChunker(self.cfg)

    def test_produces_non_empty_chunks(self):
        chunks = self.chunker.chunk(CSHARP_SOURCE, "csharp")
        self.assertGreater(len(chunks), 0)

    def test_class_extracted(self):
        chunks = self.chunker.chunk(CSHARP_SOURCE, "csharp")
        cls_names = {c.symbol_name for c in chunks if c.chunk_type == "class"}
        self.assertIn("ServiceImpl", cls_names)

    def test_interface_extracted(self):
        chunks = self.chunker.chunk(CSHARP_SOURCE, "csharp")
        cls_names = {c.symbol_name for c in chunks if c.chunk_type == "class"}
        self.assertIn("IService", cls_names)

    def test_generic_class_extracted(self):
        chunks = self.chunker.chunk(CSHARP_SOURCE, "csharp")
        cls_names = {c.symbol_name for c in chunks if c.chunk_type == "class"}
        self.assertIn("GenericRepo", cls_names)

    def test_namespace_captured(self):
        chunks = self.chunker.chunk(CSHARP_SOURCE, "csharp")
        block_names = {c.symbol_name for c in chunks if c.chunk_type == "block"}
        # Namespace symbol is the dotted name
        self.assertTrue(
            any("MyApp" in (n or "") for n in block_names),
            f"Expected namespace in {block_names}",
        )

    def test_using_directives_captured(self):
        chunks = self.chunker.chunk(CSHARP_SOURCE, "csharp")
        import_chunks = [c for c in chunks if c.chunk_type == "imports"]
        self.assertGreater(len(import_chunks), 0)

    def test_method_extracted(self):
        chunks = self.chunker.chunk(CSHARP_SOURCE, "csharp")
        fn_names = {c.symbol_name for c in chunks if c.chunk_type == "function"}
        self.assertIn("Execute", fn_names)

    def test_no_keyword_false_positives(self):
        chunks = self.chunker.chunk(CSHARP_SOURCE, "csharp")
        fn_names = {c.symbol_name for c in chunks if c.chunk_type == "function"}
        for kw in ("if", "for", "while", "return", "using"):
            self.assertNotIn(kw, fn_names)

    def test_line_ranges_valid(self):
        chunks = self.chunker.chunk(CSHARP_SOURCE, "csharp")
        for c in chunks:
            self.assertGreaterEqual(c.line_start, 1)
            self.assertGreaterEqual(c.line_end, c.line_start)


# =========================================================================
# RustChunker tests
# =========================================================================

RUST_SOURCE = '''\
use std::collections::HashMap;
use std::io;

pub struct Config {
    pub name: String,
    pub value: i32,
}

pub enum Status {
    Active,
    Inactive,
    Error(String),
}

pub trait Processor {
    fn process(&self, input: &str) -> String;
    fn reset(&mut self);
}

impl Config {
    pub fn new(name: String) -> Self {
        Config { name, value: 0 }
    }
}

pub async fn serve(addr: &str) {
    println!("Serving on {}", addr);
}

fn helper() {
    let raw = r#"this has { braces } inside"#;
    println!("{}", raw);
}

mod internal {
    pub fn inner_fn() {
        println!("inner");
    }
}
'''


class TestRustChunker(unittest.TestCase):

    def setUp(self):
        from mnemosyne.chunkers.rust_chunker import RustChunker
        self.cfg = _default_config()
        self.chunker = RustChunker(self.cfg)

    def test_produces_non_empty_chunks(self):
        chunks = self.chunker.chunk(RUST_SOURCE, "rust")
        self.assertGreater(len(chunks), 0)

    def test_fn_extracted(self):
        chunks = self.chunker.chunk(RUST_SOURCE, "rust")
        fn_names = {c.symbol_name for c in chunks if c.chunk_type == "function"}
        self.assertIn("serve", fn_names)
        self.assertIn("helper", fn_names)

    def test_pub_async_fn(self):
        chunks = self.chunker.chunk(RUST_SOURCE, "rust")
        fn_names = {c.symbol_name for c in chunks if c.chunk_type == "function"}
        self.assertIn("serve", fn_names)

    def test_struct_extracted(self):
        chunks = self.chunker.chunk(RUST_SOURCE, "rust")
        cls_names = {c.symbol_name for c in chunks if c.chunk_type == "class"}
        self.assertIn("Config", cls_names)

    def test_enum_extracted(self):
        chunks = self.chunker.chunk(RUST_SOURCE, "rust")
        cls_names = {c.symbol_name for c in chunks if c.chunk_type == "class"}
        self.assertIn("Status", cls_names)

    def test_trait_extracted(self):
        chunks = self.chunker.chunk(RUST_SOURCE, "rust")
        cls_names = {c.symbol_name for c in chunks if c.chunk_type == "class"}
        self.assertIn("Processor", cls_names)

    def test_impl_block_extracted(self):
        chunks = self.chunker.chunk(RUST_SOURCE, "rust")
        cls_names = {c.symbol_name for c in chunks if c.chunk_type == "class"}
        self.assertIn("Config", cls_names)

    def test_mod_block_extracted(self):
        chunks = self.chunker.chunk(RUST_SOURCE, "rust")
        block_names = {c.symbol_name for c in chunks if c.chunk_type == "block"}
        self.assertIn("internal", block_names)

    def test_use_statements_captured(self):
        chunks = self.chunker.chunk(RUST_SOURCE, "rust")
        import_chunks = [c for c in chunks if c.chunk_type == "imports"]
        self.assertGreater(len(import_chunks), 0)

    def test_raw_string_does_not_break_scanning(self):
        """A Rust raw string r#"..."# with braces inside should not confuse
        the brace depth scanner."""
        chunks = self.chunker.chunk(RUST_SOURCE, "rust")
        fn_chunks = [c for c in chunks if c.symbol_name == "helper"]
        self.assertEqual(len(fn_chunks), 1)
        self.assertIn("r#", fn_chunks[0].content)

    def test_line_ranges_valid(self):
        chunks = self.chunker.chunk(RUST_SOURCE, "rust")
        for c in chunks:
            self.assertGreaterEqual(c.line_start, 1)
            self.assertGreaterEqual(c.line_end, c.line_start)


# =========================================================================
# JavaChunker tests
# =========================================================================

JAVA_SOURCE = '''\
package com.example.app;

import java.util.List;
import java.util.Map;

public class UserService {

    private final String dbUrl;

    public UserService(String dbUrl) {
        this.dbUrl = dbUrl;
    }

    @Override
    public String toString() {
        return "UserService{" + dbUrl + "}";
    }

    @Deprecated
    public void oldMethod() {
        System.out.println("deprecated");
    }

    public List<String> getNames() {
        return List.of("Alice", "Bob");
    }
}

public interface Repository<T> {
    T findById(int id);
    List<T> findAll();
}

public enum Color {
    RED,
    GREEN,
    BLUE;

    public String lower() {
        return name().toLowerCase();
    }
}
'''


class TestJavaChunker(unittest.TestCase):

    def setUp(self):
        from mnemosyne.chunkers.java_chunker import JavaChunker
        self.cfg = _default_config()
        self.chunker = JavaChunker(self.cfg)

    def test_produces_non_empty_chunks(self):
        chunks = self.chunker.chunk(JAVA_SOURCE, "java")
        self.assertGreater(len(chunks), 0)

    def test_class_extracted(self):
        chunks = self.chunker.chunk(JAVA_SOURCE, "java")
        cls_names = {c.symbol_name for c in chunks if c.chunk_type == "class"}
        self.assertIn("UserService", cls_names)

    def test_interface_extracted(self):
        chunks = self.chunker.chunk(JAVA_SOURCE, "java")
        cls_names = {c.symbol_name for c in chunks if c.chunk_type == "class"}
        self.assertIn("Repository", cls_names)

    def test_enum_extracted(self):
        chunks = self.chunker.chunk(JAVA_SOURCE, "java")
        cls_names = {c.symbol_name for c in chunks if c.chunk_type == "class"}
        self.assertIn("Color", cls_names)

    def test_method_extracted(self):
        chunks = self.chunker.chunk(JAVA_SOURCE, "java")
        fn_names = {c.symbol_name for c in chunks if c.chunk_type == "function"}
        self.assertIn("toString", fn_names)

    def test_annotation_folded_into_method(self):
        """@Override annotation should be included in the toString chunk."""
        chunks = self.chunker.chunk(JAVA_SOURCE, "java")
        for c in chunks:
            if c.symbol_name == "toString" and c.chunk_type == "function":
                self.assertIn("@Override", c.content)
                break
        else:
            self.fail("toString method chunk not found")

    def test_deprecated_annotation_folded(self):
        """@Deprecated annotation should be included in the oldMethod chunk."""
        chunks = self.chunker.chunk(JAVA_SOURCE, "java")
        for c in chunks:
            if c.symbol_name == "oldMethod" and c.chunk_type == "function":
                self.assertIn("@Deprecated", c.content)
                break
        else:
            self.fail("oldMethod chunk not found")

    def test_import_captured(self):
        chunks = self.chunker.chunk(JAVA_SOURCE, "java")
        import_chunks = [c for c in chunks if c.chunk_type == "imports"]
        self.assertGreater(len(import_chunks), 0)

    def test_package_captured(self):
        chunks = self.chunker.chunk(JAVA_SOURCE, "java")
        block_chunks = [c for c in chunks if c.chunk_type == "block"]
        combined = "".join(c.content for c in block_chunks)
        self.assertIn("package", combined)

    def test_no_keyword_false_positives(self):
        chunks = self.chunker.chunk(JAVA_SOURCE, "java")
        fn_names = {c.symbol_name for c in chunks if c.chunk_type == "function"}
        for kw in ("if", "for", "while", "return", "import", "package"):
            self.assertNotIn(kw, fn_names)

    def test_line_ranges_valid(self):
        chunks = self.chunker.chunk(JAVA_SOURCE, "java")
        for c in chunks:
            self.assertGreaterEqual(c.line_start, 1)
            self.assertGreaterEqual(c.line_end, c.line_start)


# =========================================================================
# Kotlin via JavaChunker tests
# =========================================================================

KOTLIN_SOURCE = '''\
package com.example

import kotlin.math.sqrt

fun greet(name: String) {
    println("Hello, $name")
}

class Calculator {
    fun add(a: Int, b: Int): Int {
        return a + b
    }
}

object Singleton {
    val value = 42
}
'''


class TestKotlinChunker(unittest.TestCase):

    def setUp(self):
        from mnemosyne.chunkers.java_chunker import JavaChunker
        self.cfg = _default_config()
        self.chunker = JavaChunker(self.cfg)

    def test_kotlin_fun_extracted(self):
        chunks = self.chunker.chunk(KOTLIN_SOURCE, "kotlin")
        fn_names = {c.symbol_name for c in chunks if c.chunk_type == "function"}
        self.assertIn("greet", fn_names)

    def test_kotlin_class_extracted(self):
        chunks = self.chunker.chunk(KOTLIN_SOURCE, "kotlin")
        cls_names = {c.symbol_name for c in chunks if c.chunk_type == "class"}
        self.assertIn("Calculator", cls_names)

    def test_kotlin_object_extracted(self):
        chunks = self.chunker.chunk(KOTLIN_SOURCE, "kotlin")
        cls_names = {c.symbol_name for c in chunks if c.chunk_type == "class"}
        self.assertIn("Singleton", cls_names)


# =========================================================================
# Registry dispatch tests
# =========================================================================


class TestRegistryDispatch(unittest.TestCase):

    def setUp(self):
        self.cfg = _default_config()

    def test_go_dispatch(self):
        from mnemosyne.chunkers import get_chunker
        from mnemosyne.chunkers.go_chunker import GoChunker
        chunker = get_chunker("go", self.cfg)
        self.assertIsInstance(chunker, GoChunker)

    def test_csharp_dispatch(self):
        from mnemosyne.chunkers import get_chunker
        from mnemosyne.chunkers.csharp_chunker import CSharpChunker
        chunker = get_chunker("csharp", self.cfg)
        self.assertIsInstance(chunker, CSharpChunker)

    def test_rust_dispatch(self):
        from mnemosyne.chunkers import get_chunker
        from mnemosyne.chunkers.rust_chunker import RustChunker
        chunker = get_chunker("rust", self.cfg)
        self.assertIsInstance(chunker, RustChunker)

    def test_java_dispatch(self):
        from mnemosyne.chunkers import get_chunker
        from mnemosyne.chunkers.java_chunker import JavaChunker
        chunker = get_chunker("java", self.cfg)
        self.assertIsInstance(chunker, JavaChunker)

    def test_kotlin_dispatch(self):
        from mnemosyne.chunkers import get_chunker
        from mnemosyne.chunkers.java_chunker import JavaChunker
        chunker = get_chunker("kotlin", self.cfg)
        self.assertIsInstance(chunker, JavaChunker)

    def test_language_map_extensions(self):
        from mnemosyne.chunkers import detect_language
        self.assertEqual(detect_language("main.go"), "go")
        self.assertEqual(detect_language("Program.cs"), "csharp")
        self.assertEqual(detect_language("lib.rs"), "rust")
        self.assertEqual(detect_language("App.java"), "java")
        self.assertEqual(detect_language("Main.kt"), "kotlin")
        self.assertEqual(detect_language("main.cpp"), "cpp")
        self.assertEqual(detect_language("header.h"), "c")
        self.assertEqual(detect_language("header.hpp"), "cpp")

    def test_unknown_extension_still_generic(self):
        from mnemosyne.chunkers import get_chunker
        from mnemosyne.chunkers.generic_chunker import GenericChunker
        chunker = get_chunker("unknown", self.cfg)
        self.assertIsInstance(chunker, GenericChunker)


if __name__ == "__main__":
    unittest.main()
