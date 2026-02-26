use cortexa::{
    forward::Forward,
    learning::{Activation, Learning},
    network::Network,
    utils::get_dataset,
};

const EPOCHS: usize = 1;

fn main() {
    let mut network: Network = Network::builder()
        //.neurons(100)
        .layers(&[3, 4, 2])
        //.connections(20)
        .connections(&[2, 3, 1])
        .learning(Learning::Hebbian)
        .activation(Activation::Tanh)
        .build();

    let input = get_dataset(3);

    network.set_input(input);

    // Train
    //network.train();

    // More customization
    for _epoch in 0..EPOCHS {
        let output = network.forward();
        println!("{}", output);
        //let y = network.prediction();

        //network.update(y);
    }
}
