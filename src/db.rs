use std::sync::{Arc, Mutex};

#[salsa::db]
#[derive(Default, Clone)]
pub struct RippleIRDatabaseImpl {
    storage: salsa::Storage<Self>,

    logs: Arc<Mutex<Option<Vec<String>>>>,
}

#[salsa::db]
impl salsa::Database for RippleIRDatabaseImpl {
    fn salsa_event(&self, event: &dyn Fn() -> salsa::Event) {
        let event = event();
        eprintln!("Event: {event:?}");

        if let Some(logs) = &mut *self.logs.lock().unwrap() {
            if let salsa::EventKind::WillExecute { .. } = event.kind {
                logs.push(format!("Event: {event:?}"));
            }
        }
    }
}
