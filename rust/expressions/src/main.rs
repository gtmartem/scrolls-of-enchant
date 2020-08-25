fn main() {
    block_as_expression_example();

    let x: i32 = five();
    println!("Result of five() call: x = {}", x);

    println!("Result of plus_one() call: {}", plus_one(x))
}

fn five() -> i32 {
    5
}

fn plus_one(x: i32) -> i32 {
    x + 1
}

fn block_as_expression_example() {
    let x: i32 = 5;

    let y: i32 = {
        let x: i32 = 3;
        // last line in expression used without semicolon:
        x + 1
    };

    println!("Block as expression example: x = {}, y = {}", x, y)
}
