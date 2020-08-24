///
/// Multiline comment
///

fn main() {
    println!("Hello, world!");

    // another functions calls:
    another_function();
    another_function_2(5);
    another_function_3(5, 6.7);

    // vector data types printing functions calls:
    // tuple declare without annotations
    let tup = (1, 2, 3, 4, 5, 6);
    tuple_element_printer(tup);

    // tuple declare with annotations
    let tup_with_annotations: (i8, i16, i32, i64, i128, isize) = (1, 2, 3, 4, 5, 6);
    tuple_element_printer(tup_with_annotations);

    // array declaring without annotations
    let ar = [1, 2, 3, 4, 5];
    array_element_printer(ar);

    // array declaring with annotations
    let ar: [i32; 5] = [1, 2, 3, 4, 5];
    array_element_printer(ar);
}

// simple function
fn another_function() {
    println!("Another function call!");
}

// function with 1 argument and type annotation
fn another_function_2(x: i32) {
    println!("Another function #2 call with argument x = {}", x);
}

// function with two arguments of different types
fn another_function_3(x: i32, y: f32) {
    println!("Another function #3 call with arguments x = {}, y = {}", x, y);
}

// function prints first element of tuple
fn tuple_element_printer(x: (i8, i16, i32, i64, i128, isize)) {
    println!("Another function with tuple first element: {}", x.0);
}

// function prints first element of array
fn array_element_printer(x: [i32; 5]) {
    println!("Another function with array first element: {}", x[0]);
}