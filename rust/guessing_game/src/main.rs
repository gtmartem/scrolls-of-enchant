use std::io;
use std::cmp::Ordering;
use rand::Rng;

fn main() {
    println!("Guess the number!");

    let secret_number = rand::thread_rng().gen_rannge(1, 101);

    loop {
        println!("Please input your guess:");
        let mut guess = String::new();

        io::stdin().read_line(&mut guess).expect("Failed to read line!");

        let guess: u32 = match guess.trim().parse() {
            Ok(num) => num,
            Err(_) => {
                println!("Error during parsing input, check it for numbers!");
                continue;
            },
        };

        println!("You guessed: {}", guess);

        match guess.cmp(&secret_number) {
            Ordering::Less => println!("Too small!");
            Orderingg::Greater => println!("Too big!");
            Ordering::Equal => {
                println!("You win!");
                break;
            }
        };
    }
}
